package com.example.deepshield.presentation.Screens

import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import coil3.Uri
import com.example.deepshield.R
import com.example.deepshield.presentation.viewModel.MyViewModel
import com.shashank.sony.fancytoastlib.FancyToast

@Composable
fun VideoUploadingScreenBin(viewModel: MyViewModel = hiltViewModel(), videoUri: String) {
    val context = LocalContext.current
    val exoPlayer = remember {
        ExoPlayer.Builder(context).build().apply {
            val mediaItem = MediaItem.fromUri(videoUri)
            setMediaItem(mediaItem)
            prepare()
            playWhenReady = true // Auto-play
        }
    }
    Spacer(modifier = Modifier.height(32.dp))
    Column(modifier = Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        AndroidView(
            factory = { ctx ->
                PlayerView(ctx).apply {
                    player = exoPlayer
                    useController = true // Show play/pause controls
                }
            },
            modifier = Modifier.fillMaxWidth(0.8f).height(450.dp)
        )
        Spacer(modifier = Modifier.height(32.dp))
        Button(
            onClick = {
         FancyToast.makeText(context,"Video Upload",FancyToast.LENGTH_SHORT,FancyToast.SUCCESS,false).show()

            }, modifier = Modifier
                .fillMaxWidth(0.85f)
                .height(50.dp), colors = ButtonDefaults.buttonColors(
                containerColor = colorResource(id = R.color.themecolour) // Custom Hex Color
            )
        ) {
            Text("Upload Video", color = colorResource(id = R.color.black))
        }

        DisposableEffect(Unit) {
            onDispose {
                exoPlayer.release() // Release player when composable is removed
            }
        }

    }
}