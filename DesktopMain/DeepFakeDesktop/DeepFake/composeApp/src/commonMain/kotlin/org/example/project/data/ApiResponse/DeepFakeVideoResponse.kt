package org.example.project.data.ApiResponse

import kotlinx.serialization.Serializable

@Serializable
data class DeepFakeVideoResponse(
    val message: String,
    val prediction: String,
    val score: Double
)

