package com.example.deepshield.presentation.viewModel

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.deepshield.data.repoIMPL.RepositoryImpl
import com.example.deepshield.domain.StateHandling.ApiResult
import com.example.deepshield.domain.StateHandling.DeepFakeVideoResponseState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class MyViewModel @Inject constructor(private val repositoryImpl: RepositoryImpl):ViewModel() {
    private val _uploadDeepFakeVideoState= MutableStateFlow(DeepFakeVideoResponseState())
    val uploadDeepFakeVideoState= _uploadDeepFakeVideoState.asStateFlow()

    fun uploadVideoToDeepFakeServer(context: Context, videoUri: String){
        viewModelScope.launch {
            repositoryImpl.uploadVideoToDeepFakeServer(context = context, videoUri = videoUri).collectLatest {
                when(it){
                    is ApiResult.Loading->{
                        _uploadDeepFakeVideoState.value= DeepFakeVideoResponseState(isLoading = true)

                    }
                    is ApiResult.Error->{
                        _uploadDeepFakeVideoState.value= DeepFakeVideoResponseState(isLoading = false, error = it.message)
                    }
                    is ApiResult.Success->{
                        _uploadDeepFakeVideoState.value= DeepFakeVideoResponseState(isLoading = false, data = it.data)

                    }
                }
            }
        }

    }
}