package com.example.deepshield.data.UseCases.UseCaseHelper

import com.example.deepshield.data.UseCases.GetFrameFromServerUseCase
import com.example.deepshield.data.UseCases.GetGradCamUseCase
import com.example.deepshield.data.UseCases.GetHeatMapUseCase
import com.example.deepshield.data.UseCases.UploadVideoToServerUseCase

data class UseCaseHelperClass(
    val getFrameFromServerUseCase: GetFrameFromServerUseCase,
    val getHeatMapFromServerUseCase: GetHeatMapUseCase,
    val getGradCamFromServerUseCase: GetGradCamUseCase,
    val uploadVideoToDeepFakeServerUseCase: UploadVideoToServerUseCase
)
