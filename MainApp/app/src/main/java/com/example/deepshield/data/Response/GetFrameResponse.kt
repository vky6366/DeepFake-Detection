package com.example.deepshield.data.Response

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.serialization.Serializable

//data class GetFrameResponse(
//    val image_bytes: String,
//    val message: String
//)

@Serializable
data class GetFrameResponse(
    val image_bytes: List<Int>, // ✅ Matches JSON byte array format
    val message: String
) {
    // Convert List<Int> to Bitmap
    fun toBitmap(): Bitmap? {
        return try {
            // Convert List<Int> to ByteArray
            val byteArray = image_bytes.map { it.toByte() }.toByteArray()
            // Decode ByteArray to Bitmap
            BitmapFactory.decodeByteArray(byteArray, 0, byteArray.size)
        } catch (e: Exception) {
            null
        }
    }
}
