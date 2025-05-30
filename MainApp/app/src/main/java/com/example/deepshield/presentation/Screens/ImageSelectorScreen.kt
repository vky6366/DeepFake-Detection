package com.example.deepshield.presentation.Screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import coil3.compose.AsyncImage
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R
import com.example.deepshield.presentation.Utils.LoadingIndicator
import com.example.deepshield.presentation.viewModel.MyViewModel

//
//@Composable
//fun ImageSelectorScreen(navController: NavController) {
//    val imageUri= remember { mutableStateOf("") }
//    val lottiecomposition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.button))
//    val progress2 by animateLottieCompositionAsState(
//        composition = lottiecomposition,
//        iterations = LottieConstants.IterateForever,
//        speed = 0.75f
//    )
//    val medialauncher = rememberLauncherForActivityResult(
//        ActivityResultContracts.PickVisualMedia(), onResult = {uri->
//           imageUri.value = uri.toString()
//        }
//    )
//    Column(modifier = Modifier.fillMaxSize()) {
//        Text("Please Select an Image")
//        if(imageUri.value.isNullOrEmpty()){
//            Image(
//                painter = painterResource(R.drawable.logo),
//                contentDescription = "Selected Image",
//                modifier = Modifier.height(450.dp)
//            )
//        }else{
//            AsyncImage(
//                model = imageUri.value,
//                contentDescription = "Selected Image",
//                modifier = Modifier.height(450.dp),
//                placeholder = painterResource(R.drawable.logo)
//            )
//        }
//
//        Box(
//            contentAlignment = Alignment.Center,  // Centers the text inside the animation
//            modifier = Modifier
//                .fillMaxWidth(0.95f)
//                .height(50.dp)
//                .clickable {
//                    medialauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
//                }
//        ) {
//            // Lottie Animation
//            LottieAnimation(
//                composition = lottiecomposition,
//                progress = { progress2 },
//                modifier = Modifier.fillMaxWidth()  // Makes animation fill the Box
//            )
//
//            // Overlayed Text
//            Text(
//                text = "Select Video",  // Your desired text
//                color = Color.White,  // Adjust color for visibility
//                fontSize = 16.sp,
//                fontWeight = FontWeight.Bold
//            )
//        }
//
//
//    }
//
//}
@Composable
fun ImageSelectorScreen(navController: NavController,myViewModel: MyViewModel= hiltViewModel()) {
    val imageUri = remember { mutableStateOf("") }
    val imagePredictionState = myViewModel.imagePredictionState.collectAsState()
    val lottiecomposition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.button))
    val progress2 by animateLottieCompositionAsState(
        composition = lottiecomposition,
        iterations = LottieConstants.IterateForever,
        speed = 0.75f
    )
    val context = LocalContext.current
    val medialauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        imageUri.value = uri?.toString() ?: ""
    }

    val bgGradient = Brush.verticalGradient(
        colors = listOf(Color(0xFFEAF4FB), Color(0xFFD2E7F8))
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(bgGradient)

    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.Center),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(28.dp)
        ) {


            Text(
                text = "Please Select an Image",
                fontWeight = FontWeight.Bold,
                fontSize = 25.sp,
                color = Color(0xFF4A84D4)
            )

            Card(
                shape = RoundedCornerShape(20.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 12.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                modifier = Modifier
                    .fillMaxWidth(0.95f)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(320.dp),
                    contentAlignment = Alignment.Center
                ) {
                    if (imageUri.value.isEmpty()) {
                        Image(
                            painter = painterResource(R.drawable.logo),
                            contentDescription = "Selected Image",
                            modifier = Modifier.height(220.dp)
                        )
                    } else {
                        AsyncImage(
                            model = imageUri.value,
                            contentDescription = "Selected Image",
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(300.dp)
                                .clip(RoundedCornerShape(16.dp)),
                            placeholder = painterResource(R.drawable.logo),
                            contentScale = ContentScale.Crop
                        )
                    }

                }
            }

            Spacer(modifier = Modifier.height(10.dp))
            when{
                imagePredictionState.value.isLoading -> {
                    LoadingIndicator()
                }
                imagePredictionState.value.error.isNotBlank() -> {
                    Text(
                        text = imagePredictionState.value.error,
                        color = MaterialTheme.colorScheme.error
                    )
                }
                imagePredictionState.value.data != null -> {
                    Text(
                        text = imagePredictionState.value.data!!.message,
                        color = Color(0xFF4A84D4),
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = imagePredictionState.value.data!!.result,
                        color = Color(0xFF4A84D4),
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .fillMaxWidth(0.95f)
                    .height(50.dp)
                    .clickable {
                        medialauncher.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    }
            ) {
                LottieAnimation(
                    composition = lottiecomposition,
                    progress = { progress2 },
                    modifier = Modifier.fillMaxWidth()
                )
                Text(
                    text = "Select Image",
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .fillMaxWidth(0.95f)
                    .height(50.dp)
                    .clickable {
                       myViewModel.imagePrediction(context = context, imageUri = imageUri.value)
                    }
            ) {
                LottieAnimation(
                    composition = lottiecomposition,
                    progress = { progress2 },
                    modifier = Modifier.fillMaxWidth()
                )
                Text(
                    text = "Predict",
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}