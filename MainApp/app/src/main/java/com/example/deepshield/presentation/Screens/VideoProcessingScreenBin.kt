package com.example.deepshield.presentation.Screens

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import coil3.compose.rememberAsyncImagePainter
import com.example.deepshield.presentation.viewModel.MyViewModel

//@Composable
//fun VideoProcessingScreen(viewmodel:MyViewModel= hiltViewModel(),
//                         imageUri:String ,videoUri:String,
//                          navController: NavController) {
//    val deepfakeresponsestate= viewmodel.uploadDeepFakeVideoState.collectAsState()
//    val context = LocalContext.current
//    val data = remember { mutableStateOf<Bitmap?>(null) }
//    LaunchedEffect(Unit) {
//        data.value = getVideoThumbnail(context,videoUri.toUri())
//        viewmodel.uploadVideoToDeepFakeServer(context = context, videoUri = videoUri)
//
//    }
//
//    if(deepfakeresponsestate.value.data != null){
//        Log.d("APIRESPONSE", "${deepfakeresponsestate.value.data}")
//    }
//      val imageUri = imageUri.toUri() //Frame
//
//    Column(modifier = Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally,
//        verticalArrangement = Arrangement.Center)
//    {
//        Text("Implemenation")
//        data.value?.let { bitmap ->
//           Image(
//               bitmap = bitmap.asImageBitmap(),
//               contentDescription = "Video Thumbnail",
//               modifier = Modifier.size(200.dp) // Adjust the size as needed
//            )
//       }
//
//
//
//
//    }
//
//
//}
@Composable
fun VideoProcessingScreen(
    viewmodel: MyViewModel = hiltViewModel(),
    imageUri: String,
    videoUri: String,
    navController: NavController
) {
    val deepfakeResponseState = viewmodel.uploadDeepFakeVideoState.collectAsState()
    val context = LocalContext.current
    val data = remember { mutableStateOf<Bitmap?>(null) }

    LaunchedEffect(Unit) {
        Log.d("UPLOAD", "Uploading video: $videoUri")
        data.value = getVideoThumbnail(context, videoUri.toUri())
        viewmodel.uploadVideoToDeepFakeServer(context = context, videoUri = videoUri)
    }

    // Observe API Response
    LaunchedEffect(deepfakeResponseState.value) {
        deepfakeResponseState.value.data?.let { data ->
            Log.d("APIRESPONSE2", "${data.score}") // ✅ Should now print response
        }
    }
    Log.d("APIRESPONSE2", "${deepfakeResponseState.value.data?.prediction}")

    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("Implementation")

        data.value?.let { bitmap ->
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Video Thumbnail",
                modifier = Modifier.size(200.dp)
            )
        }
    }
}
