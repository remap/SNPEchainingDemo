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

class MainActivity : AppCompatActivity() {

//    companion object { init { System.loadLibrary("snpedemochaining") } }
    private var runtimeVar: Char = 'D' // prefer HTP/DSP
    private lateinit var btnRun: Button
    private lateinit var btnBuild: Button
    private lateinit var btnReBuild: Button
    private lateinit var progress: ProgressBar
    private lateinit var txtLog: TextView
    private lateinit var rg: RadioGroup

    private lateinit var helper: SNPEHelper

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnBuild = findViewById(R.id.btn_build)
        btnReBuild = findViewById(R.id.btn_rebuild)
        btnRun = findViewById(R.id.btn_run)
        progress = findViewById(R.id.progressBar)
        txtLog = findViewById(R.id.txtLog)
        rg = findViewById(R.id.rg1)

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
                    val final_tensor_name = "up_hidden0"
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