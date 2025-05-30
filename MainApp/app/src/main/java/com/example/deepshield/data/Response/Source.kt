package com.example.deepshield.data.Response

import kotlinx.serialization.Serializable

@Serializable
data class Source(
    val title: String,
    val url: String
)