//
// Created by Chiheb Boussema on 12/02/26.
//


package com.example.snpechainingdemo

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
import com.google.gson.annotations.SerializedName
import kotlin.math.log

// Data Structures
data class HFRepoInfo(val sha: String)
data class HFTreeItem(val path: String, val lfs: LFSData?)
data class LFSData(val oid: String, val size: Long)
data class SimpleManifest(val required_models: List<String>)

// The Root Manifest
data class NeuroSymbolicManifest(
    val schema_version: String,
    val suite_info: SuiteInfo,
    val environment: EnvironmentConfig,
    // We map the JSON key "models" to this list of objects
    @SerializedName("models", alternate = ["required_models"])
    val required_models: List<ModelConfig>
)

data class SuiteInfo(
    val version: String,
    val description: String,
    val min_app_version: String
)

data class EnvironmentConfig(
    val qairt_version: String,
    val dlc_version: String,
    val runtime_preference: List<String>
)

// The detailed model definition
data class ModelConfig(
    val id: String,    // e.g., "pose_estimation"
    val file: String,  // e.g., "pose_estimation_mediapipe_....dlc"
    val role: String   // e.g., "backbone"
)

data class GenAiRepo(
    val id: String,       // A unique name for local storage (e.g., "spar_pipeline")
    val hfPath: String,    // The Hugging Face path (e.g., "your-username/spar-8Elite")
    val localFolder: String
)

// Define a simple callback type
//typealias ProgressListener = (status: String, percent: Int) -> Unit
typealias ProgressListener = (status: String) -> Unit

class ModelManager(private val context: Context) {
    private val client = OkHttpClient()
    private val prefs = context.getSharedPreferences("model_integrity", Context.MODE_PRIVATE)
//    private val modelDir = context.getExternalFilesDir(null)!! // Private external storage
//    private val modelDir = context.getExternalFilesDir("dlc")!! // Private external storage

    // Secrets injected via BuildConfig
    private val hfToken = BuildConfig.HF_TOKEN
//    private val repo = BuildConfig.HF_REPO

    private val targetRepos = listOf(
        GenAiRepo("spar3d", "uclaremap/3D_gen", "spar3d/dlc"),
        GenAiRepo("SDXL", "uclaremap/SDLX_8", "dlc")
    )

    suspend fun syncAndGetPaths(onProgress: ProgressListener): Map<String, String>? {
        val validPaths = mutableMapOf<String, String>()
        val errors = mutableListOf<String>()

        val globalValidPaths = mutableMapOf<String, String>()
        val globalErrors = mutableListOf<String>()

        try {
            for (repo in targetRepos) {
                onProgress("Checking repository ${repo.id}...")
                val modelDir = context.getExternalFilesDir(repo.localFolder)!! // Private external storage
                if (!modelDir.exists()) modelDir.mkdirs() // Ensure it exists

                // 1. GLOBAL GATEKEEPER
                val remoteRepoSha = fetchRepoSha(repo.hfPath) //?: return null
//                val lastSyncedSha = prefs.getString("global_repo_sha", "")
                val lastSyncedSha = prefs.getString("sha_${repo.id}", "")
                //            val localManifest = loadLocalManifest()

                // If SHA matches, we trust our local files (Super Fast Path)
                //            if (remoteRepoSha == lastSyncedSha && verifyLocalFilesExist()) {
                //                Log.d("ModelSync", "System up to date. Skipping checks.")
                //                return getLocalPaths()
                //            }
                if ((remoteRepoSha != null && remoteRepoSha == lastSyncedSha) || (remoteRepoSha == null)) {
                    val localManifest = loadLocalManifest(modelDir, repo.id)
                    if (localManifest != null && !localManifest.required_models.isNullOrEmpty()) {
                        Log.d("ModelSync", "Verifying local assets for ${repo.id}...")
                        onProgress(if (remoteRepoSha == null) "Offline mode. Verifying local assets for ${repo.id}..." else "Verifying local assets for ${repo.id}...")
                        var allLocallyValid = true

                        // Temporary map for this check so we don't pollute the main result yet
//                        val fastPathMap = mutableMapOf<String, String>()
                        val repoFastPathMap = mutableMapOf<String, String>()

                        // Check every model in the LOCAL manifest
                        for (model in localManifest.required_models) {
//                            val filename = model.file
                            val remotePath = model.file
                            val filename = remotePath.substringAfterLast('/')
                            val file = File(modelDir, filename)//model.file)
                            // We use the cached hash from Prefs to avoid re-computing SHA-256 on every launch
                            // But we MUST check file.exists()
                            val cachedHash = prefs.getString("hash_${filename}", "") ?: ""
                            if (file.exists() && cachedHash.isNotEmpty()) {
//                                fastPathMap[model.file] = file.absolutePath
                                repoFastPathMap[filename] = file.absolutePath
                            } else {
                                Log.w(
                                    "ModelSync",
                                    "Fast Path failed for ${repo.id}: ${filename} missing or unverified."
                                )
                                allLocallyValid = false
                                break // Stop checking, we need a full sync
                            }
                        }
                        if (allLocallyValid) {
                            Log.d("ModelSync", "${repo.id} up to date and verified.")
                            onProgress("Assets verified for ${repo.id}.")
//                            return fastPathMap
                            globalValidPaths.putAll(repoFastPathMap)
                            continue
                        }
                    }
                }

                // CRITICAL: If we are offline (remoteRepoSha == null) and Fast Path failed,
                // we cannot proceed. We must fail here.
                if (remoteRepoSha == null) {
                    onProgress("Offline.")
                    Log.d(
                        "ModelSync",
                        "It seems you are offline: remoteRepoSha is null. Will not proceed with fetching manifest or model downloads."
                    )
//                    return null
                    globalErrors.add("${repo.id}: Offline and models missing.")
                    continue
                }
                Log.d("ModelSync", "Repo update detected ($remoteRepoSha). Starting sync...")
                onProgress("Syncing ${repo.id}...")

                // 2. FETCH MANIFEST & TREE METADATA
                onProgress("Fetching manifest for ${repo.id}...")
                val manifest = fetchRichManifest(repo.hfPath)
                    ?: throw IOException("Online manifest for ${repo.id} missing")
                if (manifest == null || manifest.required_models.isNullOrEmpty()) {
                    Log.e(
                        "ModelSync",
                        "Manifest for ${repo.id} parsed but 'models' list is empty or null!"
                    )
                    globalErrors.add("${repo.id}: Manifest missing or empty")
                    continue
                }
//                if (manifest.required_models.isNullOrEmpty()) {
//                    Log.e("ModelSync", "Manifest for ${repo.id} parsed but 'models' list is empty or null!")
//                    throw IOException("Manifest empty")
//                    //                return null
//                }
                // SAVE MANIFEST FOR NEXT TIME
                saveManifestLocally(modelDir,repo.id, manifest)

                // Optional: We could check manifest.suite_info.min_app_version here

                val treeItems = fetchTreeMetadata(repo.hfPath)
                    ?: emptyList() // throw IOException("Metadata missing")

                val totalModels = manifest.required_models?.size ?: 0
                // 3. VERIFICATION LOOP
                for (modelConfig in manifest.required_models) {

                    val remotePath = modelConfig.file
//                    val filename = modelConfig.file
                    val filename = remotePath.substringAfterLast('/')

                    try {
                        // Find the remote file info using the filename
                        val remoteInfo = treeItems.find { it.path == remotePath } //filename }
                            ?: throw IOException("Remote metadata not found for $remotePath")
                        val expectedHash = remoteInfo.lfs?.oid
                            ?: throw IOException("No LFS hash found for $remotePath")
                        val targetFile = File(modelDir, filename)

                        // Check Step: Do we have this file in a valid state?
                        if (!isLocallyValid(targetFile, expectedHash)) {
                            Log.d("ModelSync", "Downloading ${modelConfig.id} ($filename)...")
                            onProgress("Downloading ${modelConfig.id}...")

                            downloadAndVerify(
                                repo.hfPath,
                                remotePath,
//                                filename,
                                targetFile,
                                expectedHash,
                                onProgress
                            )

                            onProgress("Download and verification complete for ${modelConfig.id}")
//                            validPaths[filename] = targetFile.absolutePath
                            globalValidPaths[filename] = targetFile.absolutePath
                        } else { // locally valid file
                            Log.d(
                                "ModelSync",
                                "Locally valid file found for ${modelConfig.id} ($filename)..."
                            )
//                            validPaths[filename] = targetFile.absolutePath
                            globalValidPaths[filename] = targetFile.absolutePath
                        }
                    } catch (e: Exception) {
                        Log.e("ModelSync", "Failed to sync $remotePath: ${e.message}")
//                        errors.add("$filename: ${e.message}")
                        globalErrors.add("$filename: ${e.message}")
                        // CONTINUE to next iteration/model
                    }
                }
                // Only update THIS repo's SHA if its specific loop had no localized errors
                // (Optional: You could track repo-specific errors to be safer, but globalErrors works if you fail the whole process on any error)
                prefs.edit().putString("sha_${repo.id}", remoteRepoSha).apply()
            }


            if (globalErrors.isNotEmpty()) {// (errors.isNotEmpty()) {
                onProgress("Sync finished with ${globalErrors.size} errors.\n PLEASE RESTART APP!")
                Log.e("ModelSync", "Errors encountered: $globalErrors")
                //                    if (validPaths.isEmpty()) return null
                return null
            } else {
                // 4. HAND-OFF (Update Global SHA)
//                    prefs.edit().putString("global_repo_sha", remoteRepoSha).apply()
                onProgress("All models ready!")
//                    return validPaths
                return globalValidPaths
            }

            // 4. HAND-OFF (Update Global SHA)
            //                prefs.edit().putString("global_repo_sha", remoteRepoSha).apply()

            //            onProgress("Sync Complete.", 100)
            //                return getLocalPaths()

        } catch (e: Exception) {
            Log.e("ModelSync", "Sync Critical Failure: ${e.message}")
            onProgress("Error: ${e.message}")
            return null // Handle error gracefully in UI
        }
    }

    // --- HELPER LOGIC ---
//    private fun saveManifestLocally(manifest: NeuroSymbolicManifest) {
//        try {
//            val json = Gson().toJson(manifest)
//            File(modelDir, "manifest.json").writeText(json)
//        } catch (e: Exception) {
//            Log.e("ModelSync", "Failed to save local manifest: ${e.message}")
//        }
//    }

    private fun saveManifestLocally(modelDir: File, repoId: String, manifest: NeuroSymbolicManifest) {
        try {
            val json = Gson().toJson(manifest)
            File(modelDir, "manifest_${repoId}.json").writeText(json)
        } catch (e: Exception) {
            Log.e("ModelSync", "Failed to save local manifest for $repoId: ${e.message}")
        }
    }

//    private fun loadLocalManifest(): NeuroSymbolicManifest? {
//        val file = File(modelDir, "manifest.json")
//        if (!file.exists()) return null
//        return try {
//            Gson().fromJson(file.readText(), NeuroSymbolicManifest::class.java)
//        } catch (e: Exception) {
//            null // Corrupt json
//        }
//    }
    private fun loadLocalManifest(modelDir: File, repoId: String): NeuroSymbolicManifest? {
        val file = File(modelDir, "manifest_${repoId}.json")
        if (!file.exists()) return null
        return try {
            Gson().fromJson(file.readText(), NeuroSymbolicManifest::class.java)
        } catch (e: Exception) {
            null // Corrupt json
        }
    }

    private fun isLocallyValid(file: File, expectedHash: String): Boolean {
        if (!file.exists()) return false

        // Check Cache first (Fastest)
        val cachedHash = prefs.getString("hash_${file.name}", "")
        if (cachedHash == expectedHash && file.length() > 0) return true

        // Fallback: Compute Hash (Slow, but necessary if cache is missing)
        Log.d("ModelSync", "Verifying integrity of ${file.name}...")
        val calculatedHash = calculateSHA256(file)

        if (calculatedHash == expectedHash) {
            // It's valid! Update cache so we don't compute again.
            prefs.edit().putString("hash_${file.name}", calculatedHash).apply()
            return true
        }

        return false // Corrupt or old version
    }

    private fun downloadAndVerify(repo_hfPath: String, remotePath: String,
//                                  filename: String,
                                  destination: File, expectedHash: String, onProgress: ProgressListener) {
        val request = Request.Builder()
            .url("https://huggingface.co/$repo_hfPath/resolve/main/$remotePath")
            .header("Authorization", "Bearer $hfToken")
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("Download failed: ${destination.name}")

            // Setup the hasher
            val digest = java.security.MessageDigest.getInstance("SHA-256")

            // Stream to file
//            response.body!!.byteStream().use { input ->
//                FileOutputStream(destination).use { output -> input.copyTo(output) }
//            }
            // Stream: Download AND Hash simultaneously
            response.body!!.byteStream().use { rawInput ->
                // Wrap input in DigestInputStream
                java.security.DigestInputStream(rawInput, digest).use { digestInput ->
                    FileOutputStream(destination).use { output ->
                        digestInput.copyTo(output)
                    }
                }
            }

            // Get the hash immediately (No need to read the file again!)
            onProgress("Verifying integrity...") // This is now instant
            val calculatedHash = digest.digest().joinToString("") { "%02x".format(it) }

            if (calculatedHash != expectedHash) {
                destination.delete()
                throw IOException("Hash mismatch! Server: $expectedHash, Local: $calculatedHash")
            }

            // 4. Success
            prefs.edit().putString("hash_${destination.name}", calculatedHash).apply()

        }
//        // INTEGRITY CHECK (Critical Step)
//        onProgress("Verifying integrity...")
//        val downloadedHash = calculateSHA256(destination)
//        if (downloadedHash != expectedHash) {
//            destination.delete() // nuke corrupted file
//            throw IOException("Hash mismatch for $filename! Server: $expectedHash, Local: $downloadedHash")
//        }
//
//        // Save valid state
//        prefs.edit().putString("hash_$filename", downloadedHash).apply()
    }

    private fun calculateSHA256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.forEachBlock(8192) { buffer, bytes -> digest.update(buffer, 0, bytes) }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    // --- API CALLS ---

    private fun fetchRepoSha(repo_hfPath: String): String? {
        val req = Request.Builder().url("https://huggingface.co/api/models/$repo_hfPath").header("Authorization", "Bearer $hfToken").build()
        val json = client.newCall(req).execute().body?.string() ?: return null
        return Gson().fromJson(json, HFRepoInfo::class.java).sha
    }

    private fun fetchManifest(repo_hfPath: String): SimpleManifest? {
        val req = Request.Builder().url("https://huggingface.co/$repo_hfPath/resolve/main/manifest.json").header("Authorization", "Bearer $hfToken").build()
        return Gson().fromJson(client.newCall(req).execute().body?.charStream(), SimpleManifest::class.java)
    }

    private fun fetchRichManifest(repo_hfPath: String): NeuroSymbolicManifest? {
        val req = Request.Builder()
            .url("https://huggingface.co/$repo_hfPath/resolve/main/manifest.json")
            .header("Authorization", "Bearer $hfToken")
            .build()

        val response = client.newCall(req).execute()
        if (!response.isSuccessful) return null

        return Gson().fromJson(response.body?.charStream(), NeuroSymbolicManifest::class.java)
    }

    private fun fetchTreeMetadata(repo_hfPath: String): List<HFTreeItem>? {
        val req = Request.Builder().url("https://huggingface.co/api/models/$repo_hfPath/tree/main?recursive=true").header("Authorization", "Bearer $hfToken").build()
        val type = object : TypeToken<List<HFTreeItem>>() {}.type
        return Gson().fromJson(client.newCall(req).execute().body?.charStream(), type)
    }

//    private fun verifyLocalFilesExist(): Boolean = modelDir.listFiles()?.isNotEmpty() == true
//
//    private fun getLocalPaths(): Map<String, String> {
//        return modelDir.listFiles()?.associate { it.name to it.absolutePath } ?: emptyMap()
//    }
}