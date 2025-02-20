package com.example.deepshield.presentation.Navigation

import kotlinx.serialization.Serializable

@Serializable
object HOMESCREEN

@Serializable
data class DEEPFAKEVIDEOOUTPUTSCREEN(
    val message:String,
    val prediction:String,
    val score:Double
)