package com.example.deepshield.data.Response

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.serialization.Serializable

@Serializable
data class HeatMapResponse(
    val heatmap_bytes: List<Int>, // Matches JSON byte array format
    val message: String
) {
    // Convert List<Int> to Bitmap
    fun toBitmap(): Bitmap? {
        return try {
            // Convert List<Int> to ByteArray
            val byteArray = heatmap_bytes.map { it.toByte() }.toByteArray()
            // Decode ByteArray to Bitmap
            BitmapFactory.decodeByteArray(byteArray, 0, byteArray.size)
        } catch (e: Exception) {
            null
        }
    }
}

