package com.example.deepshield.domain.Repository

import android.content.Context
import android.graphics.Bitmap
import com.example.deepshield.data.Response.AudioResponse
import com.example.deepshield.data.Response.DeepFakeVideoResponse
import com.example.deepshield.data.Response.GetFrameResponse
import com.example.deepshield.data.Response.GradCamResponse
import com.example.deepshield.domain.StateHandling.ApiResult
import kotlinx.coroutines.flow.Flow

interface Repository {
    //interface class
    suspend fun uploadVideoToDeepFakeServer(context:Context,videoUrl:String): Flow<ApiResult<DeepFakeVideoResponse>>
    suspend fun getFrameFromServer():Flow<ApiResult<Bitmap>>
    suspend fun getHeatMapFromServer():Flow<ApiResult<Bitmap>>
    suspend fun getGradCamFromServer():Flow<ApiResult<GradCamResponse>>
    suspend fun uploadAudioToDeepFakeServer(context:Context,audioUrl:String):Flow<ApiResult<AudioResponse>>
}