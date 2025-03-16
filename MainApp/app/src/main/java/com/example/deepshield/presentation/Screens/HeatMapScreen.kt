package com.example.deepshield.presentation.Screens

import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R
import com.example.deepshield.presentation.Navigation.HOMESCREEN
import com.example.deepshield.presentation.Navigation.VIDEOSELECTIONSCREEN
import com.example.deepshield.presentation.viewModel.MyViewModel
import kotlinx.coroutines.delay


@Composable
fun HeatCamScreen(viewModel: MyViewModel = hiltViewModel(),navController: NavController) {
    LaunchedEffect(Unit) {
        viewModel.getHeatMapFromServer()
        delay(2000)
        viewModel.getGradCamResponse()
    }
    val getHeatMapState = viewModel.getHeatMapFromServerState.collectAsState()
    val getGradCamResponse = viewModel.getGradCamFromServerState.collectAsState()
    val lottiecomposition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.button))
    val progress2 by animateLottieCompositionAsState(
        composition = lottiecomposition,
        iterations = LottieConstants.IterateForever,
        speed = 0.75f
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFFAFAFA)) // Light background for better contrast
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        if (getHeatMapState.value.isLoading || getGradCamResponse.value.isLoading) {
            Column(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                CircularProgressIndicator(
                    color = Color(0xFF6200EE),
                    strokeWidth = 4.dp
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Loading heatmap...",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium,
                    color = Color.Gray
                )
            }
        } else if (getHeatMapState.value.data != null && getGradCamResponse.value.data != null) {
            val bitmap = getHeatMapState.value.data

            Log.d("HEATMAPRESPONSE", "Bitmap received: ${bitmap.toString()}")

            bitmap?.let {
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = "Heatmap",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp)
                        .clip(RoundedCornerShape(12.dp)) // Rounded corners
                        .border(2.dp, Color.Gray, RoundedCornerShape(12.dp))
                        .padding(8.dp),
                    contentScale = ContentScale.Fit
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            val list: List<String>? = getGradCamResponse.value.data?.focused_regions

            Text(
                text = "Important Feature Regions",
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF333333),
                modifier = Modifier.padding(bottom = 8.dp)
            )

            LazyColumn(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(8.dp)
//                    .background(Color.White, RoundedCornerShape(12.dp)) // Card-like background
//                    .border(1.dp, Color.LightGray, RoundedCornerShape(12.dp))
                    .padding(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(list!!) { data ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp),
                        elevation = CardDefaults.elevatedCardElevation(8.dp)
                    ) {
                        Text(
                            text = " $data",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Medium,
                            color = Color.DarkGray,
                            modifier = Modifier.padding(12.dp)
                        )


                    }
                }

            }
            Spacer(modifier = Modifier.height(24.dp))
            Box(
                contentAlignment = Alignment.Center,  // Centers the text inside the animation
                modifier = Modifier
                    .fillMaxWidth(0.95f)
                    .height(50.dp)
                    .clickable {
                        //HeatMap
                        navController.navigate( VIDEOSELECTIONSCREEN){
                            popUpTo(0)
                        }


                    }
            ) {
                // Lottie Animation
                LottieAnimation(
                    composition = lottiecomposition,
                    progress = { progress2 },
                    modifier = Modifier.fillMaxWidth()
                )

                // Overlayed Text
                Text(
                    text = "NewVideo",  // Your desired text
                    color = Color.White,  // Adjust color for visibility
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}
