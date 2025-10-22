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

        @JvmStatic
        external fun buildPipes(
            assetManager: android.content.res.AssetManager,
            runtimePref: Char
        ): String

        @JvmStatic external fun rebuildArbitrary(): String

        @JvmStatic
        external fun setModelDirectory(path: String)

//        @JvmStatic
//        external fun executeInference(
//            assetManager: android.content.res.AssetManager,
//            runtimePref: Char
//        ): String

        @JvmStatic
        external fun runSDXL(assetManager: android.content.res.AssetManager,
                             ids1: IntArray, ids2: IntArray) : String

//        @JvmStatic
//        external fun runSDXLWhole(assetManager: android.content.res.AssetManager,
//                             ids1: IntArray, ids2: IntArray) : String

        @JvmStatic
        external fun runSDXLWhole(assetManager: android.content.res.AssetManager,
                             ids1: IntArray, ids2: IntArray, decode_only: Boolean) : FloatArray

    }

    fun toBitmapCHW(chw: FloatArray, channels: Int, height: Int, width: Int
                    /*scale: Float = 0.5f, shift: Float = 0.5f, clamp01: Boolean = true*/): Bitmap {

//    fun toBitmapCHW(chw: FloatArray, c: Int, h: Int, w: Int,
//                    rangeNeg1to1: Boolean = true, clamp01: Boolean = true): Bitmap {
//        require(c == 3) { "toBitmapCHW expects 3 channels (RGB)" }
//        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
//        val pixels = IntArray(w * h)
//        val plane = h * w
//        val rOff = 0 * plane
//        val gOff = 1 * plane
//        val bOff = 2 * plane
//
//        for (y in 0 until h) {
//            val row = y * w
//            for (x in 0 until w) {
//                val i = row + x
//                var r = chw[rOff + i]
//                var g = chw[gOff + i]
//                var b = chw[bOff + i]
//
//                if (rangeNeg1to1) {
//                    r = 0.5f * (r + 1f)
//                    g = 0.5f * (g + 1f)
//                    b = 0.5f * (b + 1f)
//                }
//                if (clamp01) {
//                    r = r.coerceIn(0f, 1f)
//                    g = g.coerceIn(0f, 1f)
//                    b = b.coerceIn(0f, 1f)
//                }
//
//                val R = (r * 255f + 0.5f).toInt().coerceIn(0, 255)
//                val G = (g * 255f + 0.5f).toInt().coerceIn(0, 255)
//                val B = (b * 255f + 0.5f).toInt().coerceIn(0, 255)
//                pixels[i] = (0xFF shl 24) or (R shl 16) or (G shl 8) or B
//            }
//        }
//        bmp.setPixels(pixels, 0, w, 0, 0, w, h)
//        return bmp
        android.util.Log.i("IMG", "Entered Bitmap conversion function...")
        require(channels == 3)
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)
        val cStride = height * width
        val rOff = 0
        val gOff = cStride
        val bOff = 2 * cStride
        var idx = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val i = y * width + x
                var R = chw[rOff + i].toInt().coerceIn(0, 255)
                var G = chw[gOff + i].toInt().coerceIn(0, 255)
                var B = chw[bOff + i].toInt().coerceIn(0, 255)
                pixels[idx++] = (0xFF shl 24) or (R shl 16) or (G shl 8) or B
            }
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)
        return bmp
    }
}