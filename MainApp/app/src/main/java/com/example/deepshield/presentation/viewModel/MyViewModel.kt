package com.example.deepshield.presentation.viewModel

import androidx.lifecycle.ViewModel
import com.example.deepshield.data.repoIMPL.RepositoryImpl
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class MyViewModel @Inject constructor(private val repositoryImpl: RepositoryImpl):ViewModel() {

}