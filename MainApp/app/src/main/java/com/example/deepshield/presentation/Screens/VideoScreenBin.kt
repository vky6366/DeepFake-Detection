package com.example.deepshield.presentation.Screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R
import com.example.deepshield.presentation.viewModel.MyViewModel

@Composable
fun VideoScreenSelector(viewModel: MyViewModel = hiltViewModel()) {
    val videuri= remember { mutableStateOf("") }
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
            videuri.value = uri.toString()
        }
    )
    Column (modifier = Modifier.fillMaxSize() ,horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center){
        LottieAnimation(
            composition = composition,
            progress ={ progress},
            modifier = Modifier.size(250.dp).clickable {



            }

        )


    }

}