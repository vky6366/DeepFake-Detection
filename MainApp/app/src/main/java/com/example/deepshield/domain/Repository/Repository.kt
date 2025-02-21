package com.example.deepshield.domain.Repository

import android.content.Context
import com.example.deepshield.data.Response.DeepFakeVideoResponse
import com.example.deepshield.domain.StateHandling.ApiResult
import kotlinx.coroutines.flow.Flow

interface Repository {
    //interface class
    suspend fun uploadVideoToDeepFakeServer(context:Context,videoUrl:String): Flow<ApiResult<DeepFakeVideoResponse>>
}