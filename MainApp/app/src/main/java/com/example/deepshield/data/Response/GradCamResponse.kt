package com.example.deepshield.data.Response

import kotlinx.serialization.Serializable


@Serializable
data class GradCamResponse(
    val focused_regions: List<String> = emptyList()
)