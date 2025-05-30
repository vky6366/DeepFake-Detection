package com.example.deepshield.presentation.Screens

import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R

@Composable
fun  NewsChatScreen() {
    val lottiecomposition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.button))
    val progress2 by animateLottieCompositionAsState(
        composition = lottiecomposition,
        iterations = LottieConstants.IterateForever,
        speed = 0.75f
    )
    var showPredecition by remember { mutableStateOf(false) }
    val primaryBlue = Color(0xFF4A84D4) // Button/TextField border color
    val bgGradient = Brush.verticalGradient(
        colors = listOf(Color(0xFFEAF4FB), Color(0xFFD2E7F8))
    )

    var text by remember { mutableStateOf("") }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(bgGradient),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier
                .padding(16.dp)
        ) {
            // Card
            Card(
                shape = RoundedCornerShape(24.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 12.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White),
                modifier = Modifier
                    .fillMaxWidth(0.95f)
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.padding(24.dp)
                ) {
                    // Search Icon
                    if(!showPredecition){

                        Icon(
                            imageVector = Icons.Default.Search,
                            contentDescription = "Search",
                            tint = primaryBlue,
                            modifier = Modifier.size(60.dp)
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        // Title
                        Text(
                            text = "News Detector",
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Bold,
                            color = primaryBlue
                        )
                    }else{
                        Text(
                            text = "Predecition lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet ",
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Bold,
                            color = primaryBlue
                        )
                    }


                    Spacer(modifier = Modifier.height(24.dp))

                    // Input Field
                    OutlinedTextField(
                        value = text,
                        onValueChange = { text = it },
                        placeholder = { Text("Enter news topic or keyword.") },
                        shape = RoundedCornerShape(10.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = primaryBlue,
                            unfocusedBorderColor = primaryBlue,
                            cursorColor = primaryBlue,
                            focusedTextColor = Color.Black
                        ),
                        modifier = Modifier
                            .fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(24.dp))
                    Box(
                        contentAlignment = Alignment.Center,  // Centers the text inside the animation
                        modifier = Modifier
                            .fillMaxWidth(0.95f)
                            .height(50.dp)
                            .clickable {
                                showPredecition = !showPredecition


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

            Spacer(modifier = Modifier.height(24.dp))


        }
    }
}
