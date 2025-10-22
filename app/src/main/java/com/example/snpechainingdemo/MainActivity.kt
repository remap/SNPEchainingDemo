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

//import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
//import ai.djl.huggingface.tokenizers.Encoding
//
//import java.io.File
//import java.nio.file.Paths

class MainActivity : AppCompatActivity() {

//    companion object { init { System.loadLibrary("snpedemochaining") } }
    private var runtimeVar: Char = 'D' // prefer HTP/DSP
    private lateinit var btnRun: Button
    private lateinit var btnBuild: Button
    private lateinit var btnReBuild: Button
    private lateinit var btn_runTokenizers: Button
    private lateinit var btn_runEncoders: Button
    private lateinit var btn_runSDXL: Button
    private lateinit var btn_buildPipes: Button
    private lateinit var btn_runDecoder: Button
    private lateinit var progress: ProgressBar
    private lateinit var txtLog: TextView
    private lateinit var rg: RadioGroup
    private lateinit var imageView: ImageView

    private lateinit var helper: SNPEHelper

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnBuild = findViewById(R.id.btn_build)
        btnReBuild = findViewById(R.id.btn_rebuild)
        btnRun = findViewById(R.id.btn_run)
        btn_runTokenizers = findViewById(R.id.btn_runTokenizers)
        btn_runEncoders = findViewById(R.id.btn_runEncoders)
        btn_buildPipes = findViewById(R.id.btn_buildPipes)
        btn_runSDXL = findViewById(R.id.btn_runSDXL)
        btn_runDecoder = findViewById(R.id.btn_runDecoder)
        progress = findViewById(R.id.progressBar)
        txtLog = findViewById(R.id.txtLog)
        rg = findViewById(R.id.rg1)
        imageView = findViewById(R.id.imageView)

        progress.visibility = View.GONE

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
            btnRun.isEnabled = false
            btnReBuild.isEnabled = false
            btnBuild.isEnabled = false
            progress.visibility = View.VISIBLE
            txtLog.text = ""

            Thread {
                var status = ""
                try {
                    val (tok1, tok2) = TokenizerHelper.tokenize("a photo of a cute puppy")
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
                    btnRun.isEnabled = true
                    btnReBuild.isEnabled = true
                    btnBuild.isEnabled = true
                    txtLog.text = status
                }
            }.start()
        }

        btn_runEncoders.setOnClickListener {
            android.util.Log.i("Tokenizer", "Running tokenizers...")
            btnRun.isEnabled = false
            btnReBuild.isEnabled = false
            btnBuild.isEnabled = false
            btn_runTokenizers.isEnabled = false
            progress.visibility = View.VISIBLE
            txtLog.text = ""

            Thread {
                var status = ""
                try {
                    val (tok1, tok2) = TokenizerHelper.tokenize("a photo of a cute puppy")
                    android.util.Log.i("TOKENIZER", "Tokenization successful!")
                    status += SNPEHelper.runSDXL(assets, tok1.ids, tok2.ids)

                    // Later: pass tok1.ids / tok2.ids into your JNI call for the text encoder DLC
                } catch (t: Throwable) {
                    status += "\nException: ${t.message}"
                }
                android.util.Log.i("Text Encoder", status)

                runOnUiThread {
                    progress.visibility = View.GONE
                    btnRun.isEnabled = true
                    btnReBuild.isEnabled = true
                    btnBuild.isEnabled = true
                    btn_runTokenizers.isEnabled = true
                    txtLog.text = status
                }
            }.start()
        }


        helper.kt_queryRuntimes()

        // Build button
        btnBuild.setOnClickListener {
            btnBuild.isEnabled = false
            progress.visibility = View.VISIBLE
            Thread {
//                val s = SNPEHelper.buildGraph(assets, runtimeVar)
                val s = SNPEHelper.buildArbitrary(assets, runtimeVar)
                android.util.Log.i("Building: ", s)
                runOnUiThread {
//                    txtLog.text = s
                    progress.visibility = View.GONE
                    btnBuild.isEnabled = true
                }
            }.start()
        }

        // reBuild button
        btnReBuild.setOnClickListener {
            btnBuild.isEnabled = false
            btnReBuild.isEnabled = false
            progress.visibility = View.VISIBLE
            Thread {
//                val s = SNPEHelper.buildGraph(assets, runtimeVar)
                val s = SNPEHelper.rebuildArbitrary()
                android.util.Log.i("ReBuilding: ", s)
                runOnUiThread {
//                    txtLog.text = s
                    progress.visibility = View.GONE
                    btnBuild.isEnabled = true
                    btnReBuild.isEnabled = true
                }
            }.start()
        }

        btn_buildPipes.setOnClickListener {
            btnBuild.isEnabled = false
            progress.visibility = View.VISIBLE
            Thread {
//                val s = SNPEHelper.buildGraph(assets, runtimeVar)
                val s = SNPEHelper.buildPipes(assets, runtimeVar)
                android.util.Log.i("Building: ", s)
                runOnUiThread {
//                    txtLog.text = s
                    progress.visibility = View.GONE
                    btnBuild.isEnabled = true
                }
            }.start()
        }

        btnRun.setOnClickListener {
            btnRun.isEnabled = false
            progress.visibility = View.VISIBLE
            txtLog.text = ""

            Thread {
                var status = ""
                var ok = false
                try {
                    // Build & RUN the 2-model graph once (our JNI does both)
//                    status = SNPEHelper.buildTwoModelGraph(assets, runtimeVar)
                    status = SNPEHelper.runGraph()
                    // Allocate a direct buffer for "final_out" and fetch it
                    val final_tensor_name = "latent_sample" // "up_hidden0"
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
                        status += "\ngetFinalTensor ok=$ok  bytes=$bytes\n$preview"
                    }
                } catch (t: Throwable) {
                    status += "\nException: ${t.message}"
                }

                runOnUiThread {
                    progress.visibility = View.GONE
                    btnRun.isEnabled = true
                    txtLog.text = status
                    Toast.makeText(this, if (ok) "OK" else "Failed", Toast.LENGTH_SHORT).show()
                }
            }.start()
        }

        btn_runSDXL.setOnClickListener {
            btnRun.isEnabled = false
            btn_runSDXL.isEnabled = false
            btn_runTokenizers.isEnabled = false
            btn_runEncoders.isEnabled = false
            btn_buildPipes.isEnabled = false
            btnBuild.isEnabled = false
            btnReBuild.isEnabled = false
            btn_runDecoder.isEnabled = false
            progress.visibility = View.VISIBLE
            txtLog.text = ""

            Thread {
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
                    val outFloats = SNPEHelper.runSDXLWhole(assets, tok1.ids, tok2.ids, false)
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
                    btnRun.isEnabled = true
                    btn_runSDXL.isEnabled = true
                    btn_runTokenizers.isEnabled = true
                    btn_runEncoders.isEnabled = true
                    btn_buildPipes.isEnabled = true
                    btnBuild.isEnabled = true
                    btnReBuild.isEnabled = true
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
            btnRun.isEnabled = false
            btn_runSDXL.isEnabled = false
            btn_runTokenizers.isEnabled = false
            btn_runEncoders.isEnabled = false
            btn_buildPipes.isEnabled = false
            btnBuild.isEnabled = false
            btnReBuild.isEnabled = false
            btn_runDecoder.isEnabled = false
            progress.visibility = View.VISIBLE
            txtLog.text = ""

            Thread {
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
                    val outFloats = SNPEHelper.runSDXLWhole(assets, tok1.ids, tok2.ids, true)
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
                    progress.visibility = View.GONE
                    btnRun.isEnabled = true
                    btn_runSDXL.isEnabled = true
                    btn_runTokenizers.isEnabled = true
                    btn_runEncoders.isEnabled = true
                    btn_buildPipes.isEnabled = true
                    btnBuild.isEnabled = true
                    btnReBuild.isEnabled = true
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