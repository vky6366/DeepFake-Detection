package com.example.deepshield.presentation.Navigation

import android.net.Uri
import kotlinx.serialization.Serializable

@Serializable
object HOMESCREEN

@Serializable
data class DEEPFAKEVIDEOOUTPUTSCREEN(
    val message:String,
    val prediction:String,
    val score:Double
)

@Serializable
data class VIDEOUPLOADSCREENROUTE(val videoUri:String , val imageUri:String)

@Serializable
object VIDEOSELECTIONSCREEN

@Serializable
data class VIDEOPROCESSINGSCREEN(val imageUri:String , val videoUri:String)

@Serializable
object HEATMAPSCREEN

@Serializable
object SELECTDEEPFAKETYPESCREEN

@Serializable
object ALLSONGSCREEN