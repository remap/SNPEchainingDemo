package com.example.snpechainingdemo

import android.app.Application
import android.content.res.AssetManager
import android.graphics.Bitmap
import java.nio.*

class SNPEHelper(private val app: Application)  {

    external fun queryRuntimes(nativeLibDir: String): String



    fun kt_queryRuntimes() {
        val res = queryRuntimes(app.applicationInfo.nativeLibraryDir)
        android.util.Log.i("SNPE_CHAINING", res)
    }
    companion object {
        init {
            System.loadLibrary("snpechainingdemo")
        }

//        @JvmStatic
//        external fun buildTwoModelGraph(
//            assetManager: android.content.res.AssetManager,
//            runtimePref: Char
//        ): String

        @JvmStatic
        external fun getFinalTensor(dst: java.nio.ByteBuffer): Boolean

        @JvmStatic
        external fun getTensor(dst: java.nio.ByteBuffer, name: String): Boolean

        @JvmStatic external fun getTensorSizeBytes(name: String): Long

//        @JvmStatic
//        external fun buildGraph(
//            assetManager: android.content.res.AssetManager,
//            runtimePref: Char
//        ): String

        @JvmStatic external fun runGraph(): String

//        @JvmStatic
//        external fun executeInference(
//            assetManager: android.content.res.AssetManager,
//            runtimePref: Char
//        ): String

        @JvmStatic
        external fun buildArbitrary(
            assetManager: android.content.res.AssetManager,
            runtimePref: Char
        ): String

        @JvmStatic external fun rebuildArbitrary(): String

        @JvmStatic
        external fun setModelDirectory(path: String)

    }
}