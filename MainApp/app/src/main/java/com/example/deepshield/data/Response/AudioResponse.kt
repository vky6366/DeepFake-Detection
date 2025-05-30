package com.example.deepshield.data.Response

import kotlinx.serialization.Serializable

@Serializable
data class AudioResponse(
    val message: String,
    val prediction: String,
    val score: Long
)