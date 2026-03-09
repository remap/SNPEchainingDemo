//
// Created by Chiheb Boussema
//


package com.example.snpechainingdemo

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import com.example.snpechainingdemo.databinding.ActivityMainBinding
import android.view.View
import android.widget.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import android.graphics.Bitmap

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.provider.MediaStore
import android.os.Environment
import java.io.IOException
import java.io.OutputStream

import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import java.io.FileInputStream

//import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
//import ai.djl.huggingface.tokenizers.Encoding
//
//import java.io.File
//import java.nio.file.Paths

import com.google.android.gms.common.moduleinstall.ModuleInstall
import com.google.android.gms.common.moduleinstall.ModuleInstallRequest
import androidx.activity.result.contract.ActivityResultContracts
import android.graphics.ImageDecoder
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentation
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenterOptions

import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// for mediapipe segmenter
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter.ImageSegmenterOptions
import android.graphics.Color
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.framework.image.ByteBufferExtractor
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// for onnx
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession.SessionOptions

class MainActivity : AppCompatActivity() {

//    companion object { init { System.loadLibrary("snpedemochaining") } }
    private var runtimeVar: Char = 'D' // prefer HTP/DSP
//    private lateinit var btnRun: Button
//    private lateinit var btnBuild: Button
//    private lateinit var btnReBuild: Button
    private lateinit var btn_runTokenizers: Button
    private lateinit var btn_runEncoders: Button
    private lateinit var btn_runSDXL: Button
//    private lateinit var btn_buildPipes: Button
    private lateinit var btn_runDecoder: Button
    private lateinit var progress: ProgressBar
    private lateinit var txtLog: TextView
    private lateinit var rg: RadioGroup
    private lateinit var imageView: ImageView

    private lateinit var edtPrompt: EditText
    private lateinit var helper: SNPEHelper
    private lateinit var lottieAnimationView_stairs: com.airbnb.lottie.LottieAnimationView
    private lateinit var lottieAnimationView_line: com.airbnb.lottie.LottieAnimationView
    private lateinit var btn_SPAR3D: Button

    // Segmentation / background removal model
    private fun checkAndDownloadModel() {
        val options = SubjectSegmenterOptions.Builder()
            .enableForegroundBitmap()
            .build()
        val segmenter = SubjectSegmentation.getClient(options)

        val moduleInstallClient = com.google.android.gms.common.moduleinstall.ModuleInstall.getClient(this)

        // specific request for the segmenter model
        val request = com.google.android.gms.common.moduleinstall.ModuleInstallRequest.newBuilder()
            .addApi(segmenter)
            .setListener { update ->
                // FIX: Calculate progress correctly from progressInfo
                val pInfo = update.progressInfo
                val percent = if (pInfo != null && pInfo.totalBytesToDownload > 0) {
                    (pInfo.bytesDownloaded * 100 / pInfo.totalBytesToDownload).toInt()
                } else {
                    0
                }

                val state = when (update.installState) {
                    0 -> "STATE_UNKNOWN"
                    1 -> "STATE_PENDING"
                    2 -> "DOWNLOADING ($percent%)"
                    3 -> "STATE_CANCELED"
                    4 -> "STATE_COMPLETED"
                    5 -> "STATE_FAILED"
                    else -> "Unknown"
                }
                android.util.Log.i("MODEL_DOWNLOAD", "Status: $state, ErrorCode: ${update.errorCode}")

                runOnUiThread {
                    // Use 'this@MainActivity.progress' to avoid confusion with local variables
                    txtLog.text = "Model Status: $state"
                }
            }
            .build()

        moduleInstallClient.installModules(request)
            .addOnSuccessListener { response ->
                if (response.areModulesAlreadyInstalled()) {
                    android.util.Log.i("MODEL_DOWNLOAD", "Modules are already installed.")
                    txtLog.text = "Model ready!"
                } else {
                    android.util.Log.i("MODEL_DOWNLOAD", "Download started...")
                    txtLog.text = "Download started..."
                }
            }
            .addOnFailureListener { e ->
                android.util.Log.e("MODEL_DOWNLOAD", "Install request failed", e)
                txtLog.text = "Install failed: ${e.message}"
            }
    }

    // Define the Gallery Launcher
    // This handles the callback when the user picks an image
    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
//            setAppInterfaceEnabled(false)
            handleImageSelection(uri)
//            setAppInterfaceEnabled(true)
        } else {
            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showLoadingAvatar(type_of_art: String) {
        lottieAnimationView_stairs.visibility = View.GONE
        lottieAnimationView_line.visibility = View.GONE

        val animationView = when (type_of_art) {
            "line" -> lottieAnimationView_line
            "stairs" -> lottieAnimationView_stairs
            else -> lottieAnimationView_stairs
        }
        animationView.let { view ->
            view.visibility = View.VISIBLE
            view.playAnimation()
        }
    }
    private fun hideLoadingAvatar() {
        lottieAnimationView_stairs.pauseAnimation()
        lottieAnimationView_stairs.visibility = View.GONE

        lottieAnimationView_line.pauseAnimation()
        lottieAnimationView_line.visibility = View.GONE
    }

    private fun setAppInterfaceEnabled(enabled: Boolean) {
        // Disable/Enable all interactive buttons
        btn_runTokenizers.isEnabled = enabled
        btn_runEncoders.isEnabled = enabled
        btn_runSDXL.isEnabled = enabled
        btn_runDecoder.isEnabled = enabled
        btn_SPAR3D.isEnabled = enabled
        edtPrompt.isEnabled = enabled

        // Toggle Loading UI
        progress.visibility = if (enabled) View.GONE else View.VISIBLE

        // If we are enabling the app, hide the avatar, otherwise keep it or hide it based on preference
        if (enabled) hideLoadingAvatar()
    }

    private fun startStrictModelSync() {
        android.util.Log.w("Model_Sync", "========== Syncing files with cloud... =====================")
        // 1. Lock the UI immediately
        setAppInterfaceEnabled(false)
        txtLog.text = "Initializing Neuro-Symbolic Engine...\nChecking Model Integrity..."

        // 2. Launch the Async Sync Task
        lifecycleScope.launch {
            try {
                val manager = ModelManager(this@MainActivity)

                // This line BLOCKS (suspends) until the sync is 100% complete
                val modelPaths = withContext(Dispatchers.IO) {
                    manager.syncAndGetPaths { status ->
                        runOnUiThread {
                            txtLog.text = "$status"
                        }
                    }
                }

                if (modelPaths != null) {
                    // Success!
                    txtLog.text = "Models Verified. Initializing Engine..."

                    // 3. Configure SNPEHelper with the verified path
                    // Note: We use the directory from the verified paths to ensure consistency
//                    val firstPath = modelPaths.values.first()
//                    val verifiedDir = java.io.File(firstPath).parentFile?.absolutePath
////
//                    if (verifiedDir != null) {
////                        SNPEHelper.setModelDirectory(verifiedDir)
////                        android.util.Log.i("SNPE", "SNPE Directory updated to: $verifiedDir")
//                        txtLog.text = "Models Verified.\nEngine Ready."
//                    }

                    // 4. Unlock the App
                    txtLog.text = "Models Verified.\nEngine Ready."
                    setAppInterfaceEnabled(true)
                } else {
                    // Failure (Network issue or Manifest mismatch)
                    txtLog.text = "CRITICAL ERROR: Model Sync Failed.\nPlease check your internet connection and restart app."
                    // Keep UI locked or show a "Retry" button
                }
            } catch (e: Exception) {
                android.util.Log.e("Sync", "Critical Failure", e)
                txtLog.text = "Error during sync: ${e.localizedMessage}"
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

//        btnBuild = findViewById(R.id.btn_build)
//        btnReBuild = findViewById(R.id.btn_rebuild)
//        btnRun = findViewById(R.id.btn_run)
//        btn_buildPipes = findViewById(R.id.btn_buildPipes)
        btn_runTokenizers = findViewById(R.id.btn_runTokenizers)
        btn_runEncoders = findViewById(R.id.btn_runEncoders)

        btn_runSDXL = findViewById(R.id.btn_runSDXL)
        btn_runDecoder = findViewById(R.id.btn_runDecoder)
        progress = findViewById(R.id.progressBar)
        txtLog = findViewById(R.id.txtLog)
        rg = findViewById(R.id.rg1)
        imageView = findViewById(R.id.imageView)
        edtPrompt = findViewById(R.id.edtPrompt)

        progress.visibility = View.GONE
        lottieAnimationView_line = findViewById(R.id.animation_view_line_art)
        lottieAnimationView_stairs = findViewById(R.id.animation_view_stairs)

        helper = SNPEHelper(application)

        btn_SPAR3D = findViewById(R.id.btn_pick_remove_bg)

        val modelDir = getExternalFilesDir("dlc")!!.absolutePath
        SNPEHelper.setModelDirectory(modelDir)
        android.util.Log.i("SNPE", "Model dir: $modelDir")

        rg.setOnCheckedChangeListener { _, id ->
            runtimeVar = when (id) {
                R.id.CPU -> 'C'
                R.id.GPU -> 'G'
                R.id.DSP -> 'D'
                else -> 'D'
            }
        }

        // Initialize tokenizers once
        android.util.Log.i("Tokenizer", "Initializing tokenizers!")
        TokenizerHelper.init(this)
        android.util.Log.i("Tokenizer", "Tokenizers initialized!")

        // check segmentation model has been downloaded
        checkAndDownloadModel()

        // This will lock the UI, check hashes, download if needed, and THEN enable buttons.
        startStrictModelSync()

        btn_runTokenizers.setOnClickListener {
            android.util.Log.i("Tokenizer", "Running tokenizers...")
//            btnRun.isEnabled = false
//            btnReBuild.isEnabled = false
//            btnBuild.isEnabled = false
            progress.visibility = View.VISIBLE
            txtLog.text = ""

            Thread {
                var status = ""
                try {
                    val promptText = edtPrompt.text.toString().trim().ifEmpty { "a photo of a cute puppy" }
                    val (tok1, tok2) = TokenizerHelper.tokenize(promptText)

                    status += "tok1 length=${tok1.ids.size}, first5=${tok1.ids.take(15)}\n"
                    status += "tok1 attention mask=${tok1.attentionMask.take(15)}\n"

                    status += "tok2 length=${tok2.ids.size}, first5=${tok2.ids.take(15)}\n"
                    status += "tok2 attention mask=${tok2.attentionMask.take(15)}\n"

                    // Later: pass tok1.ids / tok2.ids into your JNI call for the text encoder DLC
                } catch (t: Throwable) {
                    status += "\nException: ${t.message}"
                }
                android.util.Log.i("Tokenizer", status)

                runOnUiThread {
                    progress.visibility = View.GONE
//                    btnRun.isEnabled = true
//                    btnReBuild.isEnabled = true
//                    btnBuild.isEnabled = true
                    txtLog.text = status
                }
            }.start()
        }

        btn_runEncoders.setOnClickListener {
            android.util.Log.i("Tokenizer", "Running tokenizers...")
//            btnRun.isEnabled = false
//            btnReBuild.isEnabled = false
//            btnBuild.isEnabled = false
            btn_runTokenizers.isEnabled = false
            progress.visibility = View.VISIBLE
            txtLog.text = ""

            Thread {
                var status = ""
                try {
                    val promptText = edtPrompt.text.toString().trim().ifEmpty { "a photo of a cute puppy" }
                    val (tok1, tok2) = TokenizerHelper.tokenize(promptText)
                    android.util.Log.i("TOKENIZER", "Tokenization successful!")
                    android.util.Log.i("TOKENIZER","tok1 length=${tok1.ids.size}, first8=${tok1.ids.take(15)}\n")
                    android.util.Log.i("TOKENIZER","tok2 length=${tok2.ids.size}, first8=${tok2.ids.take(15)}\n")

                    status += SNPEHelper.runSDXL(assets, tok1.ids, tok2.ids)

                    // Later: pass tok1.ids / tok2.ids into your JNI call for the text encoder DLC
                } catch (t: Throwable) {
                    status += "\nException: ${t.message}"
                }
                android.util.Log.i("Text Encoder", status)

                runOnUiThread {
                    progress.visibility = View.GONE
//                    btnRun.isEnabled = true
//                    btnReBuild.isEnabled = true
//                    btnBuild.isEnabled = true
                    btn_runTokenizers.isEnabled = true
                    txtLog.text = status
                }
            }.start()
        }


        helper.kt_queryRuntimes()

        btn_runSDXL.setOnClickListener {
            btn_runSDXL.isEnabled = false
            btn_runTokenizers.isEnabled = false
            btn_runEncoders.isEnabled = false
//            btnRun.isEnabled = false
//            btn_buildPipes.isEnabled = false
//            btnBuild.isEnabled = false
//            btnReBuild.isEnabled = false
            btn_runDecoder.isEnabled = false
            progress.visibility = View.VISIBLE
            showLoadingAvatar("line")
            txtLog.text = ""

            Thread {
                imageView.setImageDrawable(null)
                var status = ""
                var ok = false
                var bmp: Bitmap? = null
                try
                {
                    val promptText = edtPrompt.text.toString().trim().ifEmpty { "a photo of a cute puppy" }
                    android.util.Log.i("TOKENIZER", "prompt:${promptText}")
                    val (tok1, tok2) = TokenizerHelper.tokenize(promptText)
//                    val (tok1, tok2) = TokenizerHelper.tokenize("a unicorn in a fairy land")
                    android.util.Log.i("TOKENIZER", "Tokenization successful!")
                    android.util.Log.i("TOKENIZER","tok1 length=${tok1.ids.size}, first8=${tok1.ids.take(15)}\n")
                    android.util.Log.i("TOKENIZER","tok2 length=${tok2.ids.size}, first8=${tok2.ids.take(15)}\n")
//                    status += SNPEHelper.runSDXLWhole(assets, tok1.ids, tok2.ids)
                    val outFloats = SNPEHelper.runSDXLWhole(assets, tok1.ids, tok2.ids, false, false)
                    android.util.Log.i("SDXL", "Got the float image!")
                    bmp = helper.toBitmapCHW(outFloats, 3, 1024, 1024)
                    android.util.Log.i("BMP", "Got the actual image!")

//                    if (bmp == null) {
//                        Toast.makeText(this, "Inference failed", Toast.LENGTH_SHORT).show()
//                    } else {
//                        imageView.setImageBitmap(bmp)
//                    }

                    val final_tensor_name = "latents" // "up_hidden0"
//                    val bytes = SNPEHelper.getTensorSizeBytes("final_out")
                    val bytes = SNPEHelper.getTensorSizeBytes(final_tensor_name)
                    if (bytes <= 0L) {
                        android.util.Log.i("Tensor output", "Tensor empty")
                        status += "\nfinal_out size=0 ?"
                    } else {
                        val buf: ByteBuffer = ByteBuffer
                            .allocateDirect(bytes.toInt())
                            .order(ByteOrder.nativeOrder())

//                        ok = SNPEHelper.getFinalTensor(buf)
                        ok = SNPEHelper.getTensor(buf, final_tensor_name)

                        // Peek a few floats for sanity
                        val fb: FloatBuffer = buf.asFloatBuffer()
                        val preview = StringBuilder()
                        val n = minOf(8, fb.remaining())
                        val tmp = FloatArray(n)
                        fb.get(tmp, 0, n)
                        preview.append("final_out first $n floats: ${tmp.joinToString(", ")}")
                        status += "getFinalTensor ok=$ok  bytes=$bytes\n$preview"
                    }
                } catch (t: Throwable) {
                    status += "\nException: ${t.message}"
                }
                android.util.Log.i("Text Encoder", status)
                runOnUiThread {
                    progress.visibility = View.GONE
                    hideLoadingAvatar()
                    btn_runSDXL.isEnabled = true
                    btn_runTokenizers.isEnabled = true
                    btn_runEncoders.isEnabled = true
//                    btnRun.isEnabled = true
//                    btn_buildPipes.isEnabled = true
//                    btnBuild.isEnabled = true
//                    btnReBuild.isEnabled = true
                    btn_runDecoder.isEnabled = true
                    txtLog.text = status
                    Toast.makeText(this, if (ok) "OK" else "Failed", Toast.LENGTH_SHORT).show()
                    if (bmp == null) {
                        Toast.makeText(this, "Inference failed", Toast.LENGTH_SHORT).show()
                    } else {
                        imageView.setImageBitmap(bmp)
                    }
                }
            }.start()
        }

        btn_runDecoder.setOnClickListener {
            btn_runSDXL.isEnabled = false
            btn_runTokenizers.isEnabled = false
            btn_runEncoders.isEnabled = false
//            btnRun.isEnabled = false
//            btn_buildPipes.isEnabled = false
//            btnBuild.isEnabled = false
//            btnReBuild.isEnabled = false
            btn_runDecoder.isEnabled = false
//            progress.visibility = View.VISIBLE
            showLoadingAvatar("stairs")
            txtLog.text = ""

            Thread {
                imageView.setImageDrawable(null)
                var status = ""
                var ok = false
                var bmp: Bitmap? = null
                try
                {
                    val (tok1, tok2) = TokenizerHelper.tokenize("a photo of a cute puppy")
                    android.util.Log.i("TOKENIZER", "Tokenization successful!")
                    android.util.Log.i("TOKENIZER","tok1 length=${tok1.ids.size}, first8=${tok1.ids.take(15)}\n")
                    android.util.Log.i("TOKENIZER","tok2 length=${tok2.ids.size}, first8=${tok2.ids.take(15)}\n")
//                    status += SNPEHelper.runSDXLWhole(assets, tok1.ids, tok2.ids)
                    val outFloats = SNPEHelper.runSDXLWhole(assets, tok1.ids, tok2.ids, true, false)
                    android.util.Log.i("SDXL", "Got the float image!")
                    bmp = helper.toBitmapCHW(outFloats, 3, 1024, 1024)
                    android.util.Log.i("BMP", "Got the actual image!")

                    val final_tensor_name = "latents" // "up_hidden0"
//                    val bytes = SNPEHelper.getTensorSizeBytes("final_out")
                    val bytes = SNPEHelper.getTensorSizeBytes(final_tensor_name)
                    if (bytes <= 0L) {
                        android.util.Log.i("Tensor output", "Tensor empty")
                        status += "\nfinal_out size=0 ?"
                    } else {
                        val buf: ByteBuffer = ByteBuffer
                            .allocateDirect(bytes.toInt())
                            .order(ByteOrder.nativeOrder())

//                        ok = SNPEHelper.getFinalTensor(buf)
                        ok = SNPEHelper.getTensor(buf, final_tensor_name)

                        // Peek a few floats for sanity
                        val fb: FloatBuffer = buf.asFloatBuffer()
                        val preview = StringBuilder()
                        val n = minOf(8, fb.remaining())
                        val tmp = FloatArray(n)
                        fb.get(tmp, 0, n)
                        preview.append("final_out first $n floats: ${tmp.joinToString(", ")}")
                        status += "getFinalTensor ok=$ok  bytes=$bytes\n$preview"
                    }
                } catch (t: Throwable) {
                    status += "\nException: ${t.message}"
                }
                android.util.Log.i("Text Encoder", status)
                runOnUiThread {
//                    progress.visibility = View.GONE
                    hideLoadingAvatar()
                    btn_runSDXL.isEnabled = true
                    btn_runTokenizers.isEnabled = true
                    btn_runEncoders.isEnabled = true
//                    btnRun.isEnabled = true
//                    btn_buildPipes.isEnabled = true
//                    btnBuild.isEnabled = true
//                    btnReBuild.isEnabled = true
                    btn_runDecoder.isEnabled = true
                    txtLog.text = status
                    Toast.makeText(this, if (ok) "OK" else "Failed", Toast.LENGTH_SHORT).show()
                    if (bmp == null) {
                        Toast.makeText(this, "Inference failed", Toast.LENGTH_SHORT).show()
                    } else {
                        imageView.setImageBitmap(bmp)
                    }
                }
            }.start()
        }

        imageView.setOnLongClickListener {
            val bmp = (imageView.drawable as? android.graphics.drawable.BitmapDrawable)?.bitmap
            if (bmp == null) {
                Toast.makeText(this, "No image to save", Toast.LENGTH_SHORT).show()
                return@setOnLongClickListener true
            }
            // Show a simple chooser dialog
            val options = arrayOf("Save image", "Share image", "Cancel")
            androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("Image")
                .setItems(options) { dialog, which ->
                    when (which) {
                        0 -> { // Save
                            val filename = "sdxl_${System.currentTimeMillis()}.png"
                            val saved = saveBitmapToGallery(this, bmp, filename)
                            Toast.makeText(this, if (saved) "Saved as $filename" else "Save failed", Toast.LENGTH_SHORT).show()
                        }
                        1 -> { // Share
                            shareBitmap(this, bmp)
                        }
                        2 -> dialog.dismiss()
                    }
                }
                .show()
            true
        }

        btn_SPAR3D.setOnClickListener {
//            runOnUiThread { setAppInterfaceEnabled(false) }
            pickImageLauncher.launch("image/*")
//            runOnUiThread { setAppInterfaceEnabled(true) }
        }


            // Build button
//        btnBuild.setOnClickListener {
//            btnBuild.isEnabled = false
//            progress.visibility = View.VISIBLE
//            Thread {
////                val s = SNPEHelper.buildGraph(assets, runtimeVar)
//                val s = SNPEHelper.buildArbitrary(assets, runtimeVar)
//                android.util.Log.i("Building: ", s)
//                runOnUiThread {
////                    txtLog.text = s
//                    progress.visibility = View.GONE
//                    btnBuild.isEnabled = true
//                }
//            }.start()
//        }

            // reBuild button
//        btnReBuild.setOnClickListener {
//            btnBuild.isEnabled = false
//            btnReBuild.isEnabled = false
//            progress.visibility = View.VISIBLE
//            Thread {
////                val s = SNPEHelper.buildGraph(assets, runtimeVar)
//                val s = SNPEHelper.rebuildArbitrary()
//                android.util.Log.i("ReBuilding: ", s)
//                runOnUiThread {
////                    txtLog.text = s
//                    progress.visibility = View.GONE
//                    btnBuild.isEnabled = true
//                    btnReBuild.isEnabled = true
//                }
//            }.start()
//        }
//
//        btn_buildPipes.setOnClickListener {
//            btnBuild.isEnabled = false
//            progress.visibility = View.VISIBLE
//            Thread {
////                val s = SNPEHelper.buildGraph(assets, runtimeVar)
//                val s = SNPEHelper.buildPipes(assets, runtimeVar)
//                android.util.Log.i("Building: ", s)
//                runOnUiThread {
////                    txtLog.text = s
//                    progress.visibility = View.GONE
//                    btnBuild.isEnabled = true
//                }
//            }.start()
//        }
//
//        btnRun.setOnClickListener {
//            btnRun.isEnabled = false
//            progress.visibility = View.VISIBLE
//            txtLog.text = ""
//
//            Thread {
//                var status = ""
//                var ok = false
//                try {
//                    // Build & RUN the 2-model graph once (our JNI does both)
////                    status = SNPEHelper.buildTwoModelGraph(assets, runtimeVar)
//                    status = SNPEHelper.runGraph()
//                    // Allocate a direct buffer for "final_out" and fetch it
//                    val final_tensor_name = "latent_sample" // "up_hidden0"
////                    val bytes = SNPEHelper.getTensorSizeBytes("final_out")
//                    val bytes = SNPEHelper.getTensorSizeBytes(final_tensor_name)
//                    if (bytes <= 0L) {
//                        android.util.Log.i("Tensor output", "Tensor empty")
//                        status += "\nfinal_out size=0 ?"
//                    } else {
//                        val buf: ByteBuffer = ByteBuffer
//                            .allocateDirect(bytes.toInt())
//                            .order(ByteOrder.nativeOrder())
//
////                        ok = SNPEHelper.getFinalTensor(buf)
//                        ok = SNPEHelper.getTensor(buf, final_tensor_name)
//
//                        // Peek a few floats for sanity
//                        val fb: FloatBuffer = buf.asFloatBuffer()
//                        val preview = StringBuilder()
//                        val n = minOf(8, fb.remaining())
//                        val tmp = FloatArray(n)
//                        fb.get(tmp, 0, n)
//                        preview.append("final_out first $n floats: ${tmp.joinToString(", ")}")
//                        status += "\ngetFinalTensor ok=$ok  bytes=$bytes\n$preview"
//                    }
//                } catch (t: Throwable) {
//                    status += "\nException: ${t.message}"
//                }
//
//                runOnUiThread {
//                    progress.visibility = View.GONE
//                    btnRun.isEnabled = true
//                    txtLog.text = status
//                    Toast.makeText(this, if (ok) "OK" else "Failed", Toast.LENGTH_SHORT).show()
//                }
//            }.start()
//        }

    }

    fun saveBitmapToGallery(context: Context, bitmap: Bitmap, filename: String): Boolean {
        val fos: OutputStream?
        try {
            val resolver = context.contentResolver
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/png")
                // Put into Pictures/SNPEchainingDemo
                put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/SNPEchainingDemo")
            }
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
                ?: return false
            fos = resolver.openOutputStream(uri) ?: return false
            fos.use { out ->
                val ok = bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                out.flush()
                return ok
            }
        } catch (e: IOException) {
            e.printStackTrace()
            return false
        }
    }

    fun shareBitmap(context: Context, bitmap: Bitmap) {
        try {
            val cachePath = File(context.cacheDir, "images")
            cachePath.mkdirs()
            val file = File(cachePath, "shared_image.png")
            FileOutputStream(file).use { stream ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
                stream.flush()
            }

            // If you already have a FileProvider in your manifest, use it; otherwise use MediaStore
            // Assuming you added FileProvider with authority "${applicationId}.fileprovider"
            val authority = "${context.packageName}.fileprovider"
            val contentUri: Uri = try {
                FileProvider.getUriForFile(context, authority, file)
            } catch (e: IllegalArgumentException) {
                // fallback to a URI via MediaStore (less ideal), attempt insert then
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, "shared_${System.currentTimeMillis()}.png")
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/png")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/SNPEchainingDemo")
                }
                val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                context.contentResolver.openOutputStream(uri!!)?.use { os -> bitmap.compress(Bitmap.CompressFormat.PNG, 100, os) }
                uri!!
            }

            val shareIntent = Intent().apply {
                action = Intent.ACTION_SEND
                putExtra(Intent.EXTRA_STREAM, contentUri)
                type = "image/png"
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            context.startActivity(Intent.createChooser(shareIntent, "Share Image"))
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(context, "Share failed", Toast.LENGTH_SHORT).show()
        }
    }

    // 1. Load Image -> Convert RGBA -> Remove Background -> Display
    private fun handleImageSelection(uri: Uri) {
        progress.visibility = View.VISIBLE
        txtLog.text = "Processing image..."
        imageView.setImageDrawable(null) // Clear previous
//        runOnUiThread { setAppInterfaceEnabled(false) }
//        btn_runTokenizers.isEnabled = false
//        btn_runEncoders.isEnabled = false
//        btn_runSDXL.isEnabled = false
//        btn_runDecoder.isEnabled = false
//        btn_SPAR3D.isEnabled = false
//        edtPrompt.isEnabled = false


        Thread {
            try {
                // A. Load Bitmap from URI
                val originalBitmap = loadBitmapFromUri(uri)

                // B. Convert to RGBA (ARGB_8888) and ensure Mutable
                // Even if it is RGB, this forces it to RGBA
                val rgbaBitmap = if (originalBitmap.config != Bitmap.Config.ARGB_8888 || !originalBitmap.isMutable) {
                    originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
                } else {
                    originalBitmap
                }

                android.util.Log.i("IMG_PROC", "Bitmap prepared: ${rgbaBitmap.width}x${rgbaBitmap.height}, Config: ${rgbaBitmap.config}")

                // C. Run ML Kit Background Removal
                removeBackgroundMLKit(rgbaBitmap)
//                removeBackgroundMediaPipe(rgbaBitmap)
//                removeBackgroundONNX(rgbaBitmap)

            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread {
                    progress.visibility = View.GONE
                    txtLog.text = "Error loading image: ${e.message}"
                }
            }
//            runOnUiThread { setAppInterfaceEnabled(true) }
        }.start()
    }

    // Helper to safely load bitmap (Handles Android P+ vs older)
    private fun loadBitmapFromUri(uri: Uri): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.isMutableRequired = true
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
    }

    fun floatArrayToBitmapRGBA(floats: FloatArray, width: Int, height: Int): Bitmap {
        val pixels = IntArray(width * height)
        val area = width * height
        // Safety check
        if (floats.size < area * 4) {
            android.util.Log.e("IMG_PROC", "Float array too small! Expected ${area * 4}, got ${floats.size}")
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        }
        for (i in 0 until area) {
            // C++ sent us CHW planar data:
            // Plane 0 = R, Plane 1 = G, Plane 2 = B, Plane 3 = A
            // Values are 0.0f - 255.0f
            val r = floats[i].toInt().coerceIn(0, 255)
            val g = floats[i + area].toInt().coerceIn(0, 255)
            val b = floats[i + 2 * area].toInt().coerceIn(0, 255)
            val a = floats[i + 3 * area].toInt().coerceIn(0, 255)
            // Pack into Android ARGB Int: (Alpha << 24) | (Red << 16) | (Green << 8) | Blue
            pixels[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    fun moveGlbToDownloads(context: Context, sourceFile: File): android.net.Uri? {
        val resolver = context.contentResolver
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, sourceFile.name)
            put(MediaStore.MediaColumns.MIME_TYPE, "model/gltf-binary")
            // Save to "Download/SNPEchainingDemo"
//            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/SNPEchainingDemo")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOCUMENTS + "/SNPEchainingDemo")
        }

        try {
            // 1. Create the destination file in MediaStore
            val uri = resolver.insert(MediaStore.Files.getContentUri("external"), contentValues)
                ?: return null

            // 2. Copy the data
            resolver.openOutputStream(uri).use { outputStream ->
                if (outputStream == null) return null
                FileInputStream(sourceFile).use { inputStream ->
                    inputStream.copyTo(outputStream)
                }
            }

            // 3. DELETE the original source file to free up space
            // This makes it a "Move" operation
            val deleted = sourceFile.delete()
            if (!deleted) {
                android.util.Log.w("Export", "Failed to delete source file: ${sourceFile.absolutePath}")
            }

            return uri
        } catch (e: IOException) {
            e.printStackTrace()
            return null
        }
    }

    fun openGlbExternally(context: Context, fileUri: android.net.Uri) {
        val intent = android.content.Intent(android.content.Intent.ACTION_VIEW)

        // The "model/gltf-binary" MIME type is standard, but sometimes "application/octet-stream" works better for broader support
        intent.setDataAndType(fileUri, "model/gltf-binary")

        // Grant read permission to the receiving app
        intent.flags = android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION or
                       android.content.Intent.FLAG_ACTIVITY_NEW_TASK

        try {
            context.startActivity(intent)
        } catch (e: android.content.ActivityNotFoundException) {
            android.widget.Toast.makeText(context, "No 3D viewer found.", android.widget.Toast.LENGTH_SHORT).show()
        }
    }
    // Helper to run the Subject Segmenter
    private fun removeBackgroundMLKit(inputBitmap: Bitmap) {
        val options = SubjectSegmenterOptions.Builder()
            .enableForegroundBitmap()
            .build()

        val segmenter = SubjectSegmentation.getClient(options)
        val image = InputImage.fromBitmap(inputBitmap, 0)

        segmenter.process(image)
            .addOnSuccessListener { result ->
                // The result.foregroundBitmap contains the image with transparent background
                val foregroundBitmap = result.foregroundBitmap
                var showBitmap = foregroundBitmap
                if (foregroundBitmap != null) {
                    // 1. Prepare Data for JNI
                    val width = foregroundBitmap.width
                    val height = foregroundBitmap.height
                    val byteBuffer = ByteBuffer.allocateDirect(foregroundBitmap.byteCount)
                    foregroundBitmap.copyPixelsToBuffer(byteBuffer) // Copies ARGB_8888 bytes
//                    imageView.setImageBitmap(foregroundBitmap)
//                    val savePath = this.filesDir.absolutePath
                    val externalDir = this.getExternalFilesDir(null)
                    val baseDir = externalDir ?: this.filesDir
                    // Generate a timestamp string
                    val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                    val fileName = "model_$timeStamp.glb"
                    val tempFile = File(baseDir, fileName)
                    val savePath = tempFile.absolutePath
                    android.util.Log.i("savePath: ", savePath)

                    // 2. Call JNI
                    // This runs the C++ center/crop and returns the float[]
//                    val processedFloats = SNPEHelper.runSPAR3D(
//                        assets, // AssetManager
//                        byteBuffer,
//                        width,
//                        height,
//                        this.filesDir.absolutePath
//                    )
//                    // 3. Convert back to Bitmap to Verify
//                    // We know the target size is 512x512
//                    showBitmap = floatArrayToBitmapRGBA(processedFloats, 512, 512)
////                    showBitmap = floatArrayToBitmapRGBA(processedFloats, width, height)
//                }
//
//                runOnUiThread {
//                    progress.visibility = View.GONE
//                    if (foregroundBitmap != null) {
//                        imageView.setImageBitmap(showBitmap)
//                        txtLog.text = "Background removed successfully!"
//                        // Optional: If you need to send this to C++, this is where
//                        // you would call your JNI function with 'foregroundBitmap'
//                    } else {
//                        txtLog.text = "Could not detect foreground subject."
//                    }
//                }
                    lifecycleScope.launch(Dispatchers.IO) {
                        // Define the callback
                        val callback = object : SNPEHelper.PreprocessCallback {
                            override fun onPreprocessComplete(data: FloatArray) {
                                // Switch to Main to update UI immediately
                                launch(Dispatchers.Main) {
                                    val previewBitmap = floatArrayToBitmapRGBA(data, 512, 512)
                                    imageView.setImageBitmap(previewBitmap)
                                    txtLog.text = "Preprocessing done. Running Inference..."
                                }
                            }
                        }
                        // BLOCKING CALL (Runs on IO thread)
                        // The callback above will fire halfway through execution
                        val finalFloats = SNPEHelper.runSPAR3D(
                            assets,
                            byteBuffer,
                            width,
                            height,
                            savePath,
                            callback
                        )
                        if (tempFile.exists()) {
                            val savedUri = moveGlbToDownloads(this@MainActivity, tempFile)
                            if (savedUri != null) {
                                android.util.Log.i("Export", "GLB moved to Documents: $savedUri")
                                runOnUiThread {
                                    openGlbExternally(this@MainActivity, savedUri)
                                }
                            } else {
                                android.util.Log.w("Export", "Failed to move GLB to Documents")
                            }
                        }
                        // 3. Final Result Handling (Back on Main Thread)
                        withContext(Dispatchers.Main) {
                            progress.visibility = View.GONE
                            txtLog.text = "Inference Complete!"
                            // logic to save finalFloats or show 3D model...
                        }
                    }
                }
            }
            .addOnFailureListener { e ->
                runOnUiThread {
                    progress.visibility = View.GONE
                    // Check for the specific download error
                    if (e.message?.contains("Waiting for the subject segmentation") == true) {
                        txtLog.text = "Model is downloading... Please wait 1 minute and try again."
                        Toast.makeText(this, "Downloading AI Model... Please wait.", Toast.LENGTH_LONG).show()
                    } else {
                        txtLog.text = "ML Kit Error: ${e.message}"
                    }
                }
            }
    }

    private fun removeBackgroundMediaPipe(inputBitmap: Bitmap) {
    // 1. Initial UI setup
    progress.visibility = View.VISIBLE
    txtLog.text = "Initializing MediaPipe Segmenter..."

    // Launch everything on a background thread so we don't freeze the UI
    lifecycleScope.launch(Dispatchers.Default) {
        try {
            // ==========================================
            // PART A: MEDIAPIPE SEGMENTATION
            // ==========================================
            txtLog.text = "0"
            // 1. Setup options (Ensure "deeplabv3.tflite" is in your src/main/assets/ folder)
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("deeplab_v3.tflite")
                .build()
            txtLog.text = "1"

            val options = ImageSegmenter.ImageSegmenterOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setOutputCategoryMask(true)
                .setOutputConfidenceMasks(false)
                .build()
            txtLog.text = "2"

            // 2. Create Segmenter
            val segmenter = ImageSegmenter.createFromOptions(this@MainActivity, options)
            txtLog.text = "3"

            withContext(Dispatchers.Main) { txtLog.text = "Segmenting Image..." }

            // 3. Process the image (Blocking call)
            val mpImage = BitmapImageBuilder(inputBitmap).build()
            val result = segmenter.segment(mpImage)

            val masks = result.categoryMask()
            if (!masks.isPresent) {
                throw Exception("No mask returned from MediaPipe.")
            }

            // 4. Extract the mask bytes
            val maskImage = masks.get()
            val maskBuffer = ByteBufferExtractor.extract(maskImage)
            maskBuffer.rewind()

            val width = inputBitmap.width
            val height = inputBitmap.height

            // Get original pixels
            val inputPixels = IntArray(width * height)
            inputBitmap.getPixels(inputPixels, 0, width, 0, 0, width, height)

            // Prepare output pixels
            val outputPixels = IntArray(width * height)

            // Apply mask: if category is 0, it's background. If > 0, it's an object.
            for (i in 0 until width * height) {
                val category = maskBuffer.get().toInt() and 0xFF
                if (category > 0) { // Keep Foreground
                    outputPixels[i] = inputPixels[i]
                } else {            // Erase Background
                    outputPixels[i] = Color.TRANSPARENT
                }
            }

            // 5. Create the transparent foreground bitmap
            val foregroundBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            foregroundBitmap.setPixels(outputPixels, 0, width, 0, 0, width, height)

            // Free MediaPipe memory
            segmenter.close()

            withContext(Dispatchers.Main) { txtLog.text = "Preprocessing for 3D..." }

            // ==========================================
            // PART B: C++ JNI PIPELINE (Your existing code)
            // ==========================================

            val byteBuffer = ByteBuffer.allocateDirect(foregroundBitmap.byteCount)
            foregroundBitmap.copyPixelsToBuffer(byteBuffer) // Copies ARGB_8888 bytes

            val externalDir = getExternalFilesDir(null)
            val baseDir = externalDir ?: filesDir
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val tempFile = File(baseDir, "model_$timeStamp.glb")
            val savePath = tempFile.absolutePath

            android.util.Log.i("savePath: ", savePath)

            // Define the callback
            val callback = object : SNPEHelper.PreprocessCallback {
                override fun onPreprocessComplete(data: FloatArray) {
                    // Switch to Main to update UI immediately
                    lifecycleScope.launch(Dispatchers.Main) {
                        val previewBitmap = floatArrayToBitmapRGBA(data, 512, 512)
                        imageView.setImageBitmap(previewBitmap)
                        txtLog.text = "Preprocessing done. Running Inference..."
                    }
                }
            }

            // BLOCKING C++ CALL
            val finalFloats = SNPEHelper.runSPAR3D(
                assets,
                byteBuffer,
                width,
                height,
                savePath,
                callback
            )

            // Move the file and open it
            if (tempFile.exists()) {
                val savedUri = moveGlbToDownloads(this@MainActivity, tempFile)
                if (savedUri != null) {
                    android.util.Log.i("Export", "GLB moved to Documents: $savedUri")
                    withContext(Dispatchers.Main) {
                        openGlbExternally(this@MainActivity, savedUri)
                    }
                } else {
                    android.util.Log.w("Export", "Failed to move GLB to Documents")
                }
            }

            // Final Result Handling
            withContext(Dispatchers.Main) {
                progress.visibility = View.GONE
                txtLog.text = "Inference Complete!"
            }

        } catch (e: Exception) {
            e.printStackTrace()
            withContext(Dispatchers.Main) {
                progress.visibility = View.GONE
                txtLog.text = "Error: ${e.message}"
            }
        }
    }
}

    private fun getAssetFilePath(assetName: String): String {
        val file = File(cacheDir, assetName)
        // Only copy it if it doesn't already exist
        if (!file.exists()) {
            assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        return file.absolutePath
    }
    private fun removeBackgroundONNX(inputBitmap: Bitmap) {
        progress.visibility = View.VISIBLE
        txtLog.text = "Initializing ONNX Engine..."

        lifecycleScope.launch(Dispatchers.Default) {
            var env: OrtEnvironment? = null
            try {
                // ==========================================
                // PART A: ONNX INFERENCE (RMBG / U^2-Net)
                // ==========================================
                env = OrtEnvironment.getEnvironment()

                // 1. Load the model from assets
//                val session = env.createSession(assets.open("model.onnx").readBytes())
                val modelPath = getAssetFilePath("model.onnx")

                val options = SessionOptions()
                options.addNnapi() // Tells ONNX to use Android's hardware accelerator

//                val session = env.createSession(modelPath)
                val session = env.createSession(modelPath, options)

                val inputName = session.inputNames.iterator().next() // usually "input" or "image"

                // 2. Prepare Image (Most high-quality matting models expect 1024x1024)
                val targetSize = 1024
                val resizedBitmap = Bitmap.createScaledBitmap(inputBitmap, targetSize, targetSize, true)

                withContext(Dispatchers.Main) { txtLog.text = "Extracting Alpha Mask..." }

                // 3. Convert Bitmap to CHW FloatBuffer (Normalized to -0.5 to 0.5)
                val imgData = FloatBuffer.allocate(3 * targetSize * targetSize)
                val pixels = IntArray(targetSize * targetSize)
                resizedBitmap.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)

                val rOffset = 0
                val gOffset = targetSize * targetSize
                val bOffset = 2 * targetSize * targetSize

                // Unroll channels into flat array for the tensor
                for (i in pixels.indices) {
                    val color = pixels[i]
                    // Bria RMBG standard normalization: (pixel / 255.0) - 0.5
                    imgData.put(rOffset + i, (Color.red(color) / 255.0f) - 0.5f)
                    imgData.put(gOffset + i, (Color.green(color) / 255.0f) - 0.5f)
                    imgData.put(bOffset + i, (Color.blue(color) / 255.0f) - 0.5f)
                }
                imgData.rewind() // Reset buffer position before reading!

                // 4. Run Inference
                val shape = longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong())
                val inputTensor = OnnxTensor.createTensor(env, imgData, shape)
                withContext(Dispatchers.Main) { txtLog.text = "Running session..." }
                val result = session.run(mapOf(inputName to inputTensor))

                withContext(Dispatchers.Main) { txtLog.text = "Processing results..." }
                // 5. Get the Output Alpha Mask [1, 1, 1024, 1024]
                val outputTensor = result[0] as OnnxTensor
                val alphaBuffer = outputTensor.floatBuffer

                // 6. Merge RGB + Alpha into a final Transparent RGBA Bitmap
                val finalPixels = IntArray(targetSize * targetSize)
                for (i in pixels.indices) {
                    val originalColor = pixels[i]

                    // Read alpha value from model. Usually 0.0 to 1.0.
                    var alphaFloat = alphaBuffer.get(i)

                    // If the model exports raw logits instead of sigmoids, apply sigmoid:
                    // alphaFloat = (1.0f / (1.0f + kotlin.math.exp(-alphaFloat)))

                    // Clamp just to be safe
                    if (alphaFloat < 0f) alphaFloat = 0f
                    if (alphaFloat > 1f) alphaFloat = 1f

                    val alphaInt = (alphaFloat * 255).toInt()

                    // Compose ARGB_8888
                    finalPixels[i] = Color.argb(
                        alphaInt,
                        Color.red(originalColor),
                        Color.green(originalColor),
                        Color.blue(originalColor)
                    )
                }

                val foregroundBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
                foregroundBitmap.setPixels(finalPixels, 0, targetSize, 0, 0, targetSize, targetSize)

                // Cleanup ONNX Memory
                inputTensor.close()
                result.close()
                session.close()

                withContext(Dispatchers.Main) { txtLog.text = "Preprocessing for 3D..." }

                // ==========================================
                // PART B: C++ JNI PIPELINE
                // ==========================================

                // Extract the perfect RGBA bytes
                val byteBuffer = ByteBuffer.allocateDirect(foregroundBitmap.byteCount)
                foregroundBitmap.copyPixelsToBuffer(byteBuffer)

                val externalDir = getExternalFilesDir(null)
                val baseDir = externalDir ?: filesDir
                val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val tempFile = File(baseDir, "model_$timeStamp.glb")
                val savePath = tempFile.absolutePath

                // Callback for halfway UI update
                val callback = object : SNPEHelper.PreprocessCallback {
                    override fun onPreprocessComplete(data: FloatArray) {
                        lifecycleScope.launch(Dispatchers.Main) {
                            val previewBitmap = floatArrayToBitmapRGBA(data, 512, 512)
                            imageView.setImageBitmap(previewBitmap)
                            txtLog.text = "Preprocessing done. Running Inference..."
                        }
                    }
                }

                // Blocking JNI Call (Feeding it our gorgeous 1024x1024 transparent image)
                val finalFloats = SNPEHelper.runSPAR3D(
                    assets,
                    byteBuffer,
                    targetSize, // 1024
                    targetSize, // 1024
                    savePath,
                    callback
                )

                // Move the file and open it
                if (tempFile.exists()) {
                    val savedUri = moveGlbToDownloads(this@MainActivity, tempFile)
                    if (savedUri != null) {
                        withContext(Dispatchers.Main) { openGlbExternally(this@MainActivity, savedUri) }
                    }
                }

                withContext(Dispatchers.Main) {
                    progress.visibility = View.GONE
                    txtLog.text = "Inference Complete!"
                }

            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    progress.visibility = View.GONE
                    txtLog.text = "ONNX Error: ${e.message}"
                }
            } finally {
                env?.close() // Prevent memory leaks
            }
        }
    }

}