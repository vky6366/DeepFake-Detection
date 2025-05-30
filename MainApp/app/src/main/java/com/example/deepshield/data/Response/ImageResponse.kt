package com.example.deepshield.data.Response

import kotlinx.serialization.Serializable

@Serializable
data class ImageResponse(
    val confidence: Double,
    val message: String,
    val result: String,
    val status_code: Int
)