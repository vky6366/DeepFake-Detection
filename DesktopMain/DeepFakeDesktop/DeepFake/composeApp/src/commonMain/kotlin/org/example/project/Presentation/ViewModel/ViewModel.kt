package org.example.project.Presentation.ViewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import org.example.project.data.stateHandler.ApiResult
import org.example.project.data.stateHandler.DeepFakeVideoResponseState
import org.example.project.domain.UseCase.UploadVideoToDeepFakeServerUseCase


class MyViewModel(private val uploadVideoToDeepFakeServerUseCase: UploadVideoToDeepFakeServerUseCase): ViewModel() {

    private val _uploadDeepFakeVideoState = MutableStateFlow(DeepFakeVideoResponseState())
    val uploadDeepFakeVideoState = _uploadDeepFakeVideoState.asStateFlow()

    // same for others...

    fun uploadVideoToDeepFakeServer(videoBytes: ByteArray) {
        viewModelScope.launch {
            uploadVideoToDeepFakeServerUseCase.invoke(videoBytes).collectLatest {
                when (it) {
                    is ApiResult.Loading -> {
                        _uploadDeepFakeVideoState.value = DeepFakeVideoResponseState(isLoading = true)
                    }
                    is ApiResult.Error -> {
                        _uploadDeepFakeVideoState.value = DeepFakeVideoResponseState(isLoading = false, error = it.message)
                    }
                    is ApiResult.Success -> {
                        _uploadDeepFakeVideoState.value = DeepFakeVideoResponseState(isLoading = false, data = it.data)
                    }
                }
            }
        }
    }

}
