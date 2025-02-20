package com.example.deepshield.presentation.Screens

import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.hilt.navigation.compose.hiltViewModel
import com.example.deepshield.presentation.viewModel.MyViewModel

@Composable
fun DeepFakeVideoScreen(viewmodel: MyViewModel= hiltViewModel()){
    val uploadVideoState= viewmodel.uploadDeepFakeVideoState.collectAsState()
    Log.d("APIRESPONSE","${uploadVideoState.value.data}")
    Column(modifier = Modifier.fillMaxSize()) {
        val videurl= remember { mutableStateOf("") }
        val mediapicker= rememberLauncherForActivityResult(
            ActivityResultContracts.PickVisualMedia(), onResult = {
                videurl.value=it.toString()
            }
        )
        val context = LocalContext.current
        Button(onClick = {
            mediapicker.launch(
                PickVisualMediaRequest(
                    ActivityResultContracts.PickVisualMedia.VideoOnly
                ))



        }) {
            Text("Video Input")
        }
        Button(onClick = {
            if(videurl.value.isNotEmpty()){
                viewmodel.uploadVideoToDeepFakeServer(context = context, videoUri = videurl.value)
            }else{
                Toast.makeText(context, "Error", Toast.LENGTH_SHORT).show()
            }
        }) {
            Text("Upload")
        }
    }

}