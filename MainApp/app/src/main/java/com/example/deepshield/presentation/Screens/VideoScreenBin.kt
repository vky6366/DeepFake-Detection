package com.example.deepshield.presentation.Screens

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.semantics.SemanticsProperties.TestTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.net.toUri
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R
import com.example.deepshield.data.Constants.TestTags
import com.example.deepshield.presentation.Navigation.VIDEOUPLOADSCREENROUTE
import com.example.deepshield.presentation.viewModel.MyViewModel
import com.shashank.sony.fancytoastlib.FancyToast

@Composable
fun VideoScreenSelector(viewmodel: MyViewModel = hiltViewModel(), navController: NavController) {
    val videouri= remember { mutableStateOf("") }
    val data = remember { mutableStateOf<Bitmap?>(null) }
    val navigationFlag = remember { mutableStateOf(false) }
    val lottiecomposition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.button))
    val progress2 by animateLottieCompositionAsState(
        composition = lottiecomposition,
        iterations = LottieConstants.IterateForever,
        speed = 0.75f
    )
    val context = LocalContext.current
    val composition by rememberLottieComposition(
        LottieCompositionSpec.RawRes(R.raw.cloud)
    )
    val progress by animateLottieCompositionAsState(
        composition = composition,
        iterations = LottieConstants.IterateForever,
        speed = 0.85f
    )
    val medialauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(), onResult = {uri->
            videouri.value = uri.toString()
            navigationFlag.value = false
        }
    )
    LaunchedEffect(videouri.value) {

        if (videouri.value.isNotEmpty() && !navigationFlag.value) {
            navigationFlag.value = true

            Log.d("VideoThumbnail", "$data")
            FancyToast.makeText(
                context,
                "Video Selected !",
                FancyToast.LENGTH_SHORT,
                FancyToast.SUCCESS,
                false
            ).show()

            navController.navigate(
                VIDEOUPLOADSCREENROUTE(
                    videoUri = videouri.value,
                    imageUri = data.value.toString()
                )
            )

        } else {
            FancyToast.makeText(
                context,
                "Select video",
                FancyToast.LENGTH_SHORT,
                FancyToast.WARNING,
                false
            ).show()
        }
    }
    Column (modifier = Modifier.fillMaxSize() ,horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center){
        LottieAnimation(
            composition = composition,
            progress ={ progress},
            modifier = Modifier.size(250.dp).clickable {

                medialauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))


            }

        )
        Spacer(modifier = Modifier.height(16.dp))
        // Upload Button
//        Button(
//            onClick = {
//                medialauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
//
//            }, modifier = Modifier
//                .fillMaxWidth(0.85f)
//                .height(50.dp), colors = ButtonDefaults.buttonColors(
//                containerColor = colorResource(id = R.color.themecolour) // Custom Hex Color
//            )
//        ) {
//            Text("Select Video", color = colorResource(id = R.color.black))
//        }
        //end

        Box(
            contentAlignment = Alignment.Center,  // Centers the text inside the animation
            modifier = Modifier
                .fillMaxWidth(0.95f)
                .height(50.dp)
                .testTag(TestTags.VIDEOSELECTIONSCREEN)
                .clickable {
                    medialauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
                }
        ) {
            // Lottie Animation
            LottieAnimation(
                composition = lottiecomposition,
                progress = { progress2 },
                modifier = Modifier.fillMaxWidth()  // Makes animation fill the Box
            )

            // Overlayed Text
            Text(
                text = "Select Video",  // Your desired text
                color = Color.White,  // Adjust color for visibility
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
        }



    }

}
fun getVideoThumbnail(context: Context, videoUri: Uri): Bitmap? {
    val retriever = MediaMetadataRetriever()
    return try {
        retriever.setDataSource(context, videoUri) // Set video URI
        val frameTime = 1_000_000L // Time in microseconds (1 second)

        val bitmap = retriever.getFrameAtTime(frameTime, MediaMetadataRetriever.OPTION_CLOSEST) // Extract frame

        if (bitmap != null) {
            Log.d("VideoThumbnail", "Thumbnail extracted successfully")
        } else {
            Log.e("VideoThumbnail", "Failed to extract thumbnail")
        }

        bitmap
    } catch (e: Exception) {
        Log.e("VideoThumbnail", "Error extracting thumbnail: ${e.message}", e)
        null
    } finally {
        retriever.release() // Ensure resources are released
    }
}

//        data.value?.let { bitmap ->
//            Image(
//                bitmap = bitmap.asImageBitmap(),
//                contentDescription = "Video Thumbnail",
//                modifier = Modifier.size(200.dp) // Adjust the size as needed
//            )
//        }