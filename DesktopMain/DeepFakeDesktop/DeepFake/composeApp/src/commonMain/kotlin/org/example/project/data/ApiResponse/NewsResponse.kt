package org.example.project.data.ApiResponse

import kotlinx.serialization.Serializable


@Serializable
data class NewResponse(
    val claim: String,
    val result: String,
    val similarity_score: Double,
    val sources: List<Source>
)

@Serializable
data class Source(
    val title: String,
    val url: String
)