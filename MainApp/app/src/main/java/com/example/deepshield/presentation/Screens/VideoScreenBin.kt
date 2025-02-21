package com.example.deepshield.presentation.Screens

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R
import com.example.deepshield.presentation.Navigation.VIDEOUPLOADSCREENROUTE
import com.example.deepshield.presentation.viewModel.MyViewModel
import com.shashank.sony.fancytoastlib.FancyToast

@Composable
fun VideoScreenSelector(viewmodel: MyViewModel = hiltViewModel(), navController: NavController) {
    val videouri= remember { mutableStateOf("") }
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
        }
    )
    if(videouri.value.isNotEmpty()){
        FancyToast.makeText(context,"Video Selected !",FancyToast.LENGTH_SHORT,FancyToast.SUCCESS,false).show()
        navController.navigate(VIDEOUPLOADSCREENROUTE(videoUri = videouri.value))
    }else{
        FancyToast.makeText(context,"Select video",FancyToast.LENGTH_SHORT,FancyToast.WARNING,false).show()
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
        Button(
            onClick = {
                medialauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))



            }, modifier = Modifier
                .fillMaxWidth(0.85f)
                .height(50.dp), colors = ButtonDefaults.buttonColors(
                containerColor = colorResource(id = R.color.themecolour) // Custom Hex Color
            )
        ) {
            Text("Select Video", color = colorResource(id = R.color.black))
        }


    }

}