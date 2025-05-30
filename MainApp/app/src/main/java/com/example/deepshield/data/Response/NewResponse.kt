package com.example.deepshield.data.Response

import kotlinx.serialization.Serializable

@Serializable
data class NewResponse(
    val claim: String,
    val result: String,
    val similarity_score: Double,
    val sources: List<Source>
)