package com.example.snpechainingdemo

import android.content.Context
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.huggingface.tokenizers.Encoding
import java.io.File
import java.nio.file.Paths

data class TokenMeta(
    val maxLen: Int = 77,
    val paddinSide: String = "right",
    val padId: Int?,
)
object TokenizerHelper {
    private var tok1: HuggingFaceTokenizer? = null
    private var tok2: HuggingFaceTokenizer? = null
    private var tok1_meta: TokenMeta? = null
    private var tok2_meta: TokenMeta? = null

    fun init(context: Context) {
        tok1_meta = TokenMeta(77, "right", 49407)
        tok2_meta = TokenMeta(77, "right", 0)

        tok1 = buildTokenizerFromAsset(context,
            "tokenizers/tok1/tokenizer.json",
            77,
//            "right",
//            49407
        )
        tok2 = buildTokenizerFromAsset(context,
            "tokenizers/tok2/tokenizer.json",
            77,
//            "right",
//            0
        )
    }

    private fun buildTokenizerFromAsset(ctx: Context,
                                        assetPath: String,
                                        maxLen: Int,
//                                        paddingSide: String,
//                                        padId: Int
    ): HuggingFaceTokenizer {
        val localFile = copyAssetOnce(ctx, assetPath)
        return HuggingFaceTokenizer.builder()
            .optTokenizerPath(Paths.get(localFile.absolutePath))
            .optAddSpecialTokens(true)
            .optTruncation(true)
            .optPadding(true)
            .optPadToMaxLength()
            .optMaxLength(maxLen)
            .build()
    }

    private fun copyAssetOnce(ctx: Context, assetPath: String): File {
        val outFile = File(ctx.filesDir, assetPath.substringAfterLast('/'))
        if (!outFile.exists()) {
            ctx.assets.open(assetPath).use { input ->
                outFile.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return outFile
    }

    data class Tokens(
        val ids: IntArray,
        val attentionMask: IntArray
    )

    fun tokenize(prompt: String): Pair<Tokens, Tokens> {
        val t1 = tok1 ?: error("Tokenizer 1 not initialized")
        val t2 = tok2 ?: error("Tokenizer 2 not initialized")

        val e1: Encoding = t1.encode(prompt)
        val e2: Encoding = t2.encode(prompt)

        val ids1 = e1.ids.map { it.toInt() }.toIntArray()
        val mask1 = e1.attentionMask.map { it.toInt() }.toIntArray()
        val ids2 = e2.ids.map { it.toInt() }.toIntArray()
        val mask2 = e2.attentionMask.map { it.toInt() }.toIntArray()

        // Fix padding: tok1 should pad with EOS (49407), tok2 pads with 0
        replacePad(ids1, mask1, padId = tok1_meta?.padId ?: 49407)
        replacePad(ids2, mask2, padId = tok2_meta?.padId ?: 0)

        return Tokens(ids1, mask1) to Tokens(ids2, mask2)
    }

    private fun replacePad(ids: IntArray, mask: IntArray, padId: Int) {
        for (i in ids.indices) {
            if (mask[i] == 0) {
                ids[i] = padId
            }
        }
    }
}