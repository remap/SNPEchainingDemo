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
import java.io.IOException
import java.io.OutputStream

import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream

//import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
//import ai.djl.huggingface.tokenizers.Encoding
//
//import java.io.File
//import java.nio.file.Paths

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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

//        btnBuild = findViewById(R.id.btn_build)
//        btnReBuild = findViewById(R.id.btn_rebuild)
//        btnRun = findViewById(R.id.btn_run)
        btn_runTokenizers = findViewById(R.id.btn_runTokenizers)
        btn_runEncoders = findViewById(R.id.btn_runEncoders)
//        btn_buildPipes = findViewById(R.id.btn_buildPipes)
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

}

//    private lateinit var binding: ActivityMainBinding
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//
//        binding = ActivityMainBinding.inflate(layoutInflater)
//        setContentView(binding.root)
//
//        // Example of a call to a native method
//        binding.sampleText.text = stringFromJNI()
//    }
//
//    /**
//     * A native method that is implemented by the 'snpechainingdemo' native library,
//     * which is packaged with this application.
//     */
//    external fun stringFromJNI(): String
//
//    companion object {
//        // Used to load the 'snpechainingdemo' library on application startup.
//        init {
//            System.loadLibrary("snpechainingdemo")
//        }
//    }
//}