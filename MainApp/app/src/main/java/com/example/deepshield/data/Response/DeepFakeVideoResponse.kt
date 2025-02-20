package com.example.deepshield.data.Response

data class DeepFakeVideoResponse(
    val message: String,
    val prediction: String,
    val score: Double
)