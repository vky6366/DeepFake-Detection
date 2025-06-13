package com.example.deepshield.presentation.Screens

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
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
import com.example.deepshield.presentation.Utils.LoadingIndicator
import com.example.deepshield.presentation.viewModel.MyViewModel
import com.shashank.sony.fancytoastlib.FancyToast

@Composable
fun  NewsChatScreen(viewmodel: MyViewModel= hiltViewModel(),navController: NavController) {
    val lottiecomposition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.button))
    val progress2 by animateLottieCompositionAsState(
        composition = lottiecomposition,
        iterations = LottieConstants.IterateForever,
        speed = 0.75f
    )
    val context = LocalContext.current
    var showPredecition by remember { mutableStateOf(false) }
    val primaryBlue = Color(0xFF4A84D4) // Button/TextField border color
    val bgGradient = Brush.verticalGradient(
        colors = listOf(Color(0xFFEAF4FB), Color(0xFFD2E7F8))
    )
    var text by remember { mutableStateOf("") }
    val newPredecitionState = viewmodel.newPredictionState.collectAsState()


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
            when{
                newPredecitionState.value.isLoading -> {
                    LoadingIndicator()
                }
                newPredecitionState.value.data != null -> {
                    val data = newPredecitionState.value.data!!
                    Card(
                        modifier = Modifier
                            .fillMaxWidth(0.95f)
                            .padding(bottom = 16.dp),
                        shape = RoundedCornerShape(18.dp),
                        elevation = CardDefaults.cardElevation(defaultElevation = 10.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFFEAF4FB))
                    ) {
                        LazyColumn(modifier = Modifier.padding(18.dp)) {
                            item {
                                // Result Label
                                Text(
                                    text = "Result: ${data.result}",
                                    fontWeight = FontWeight.Bold,
                                    color = if (data.result.equals(
                                            "Real",
                                            ignoreCase = true
                                        )
                                    ) Color(0xFF4CAF50) else Color(0xFFD32F2F),
                                    fontSize = 20.sp
                                )
                                Spacer(modifier = Modifier.height(8.dp))

                                // Similarity Score
                                Text(
                                    text = "Similarity Score: ${data.similarity_score}",
                                    fontSize = 15.sp,
                                    color = Color.Gray
                                )
                                Spacer(modifier = Modifier.height(8.dp))

                                // Claim
                                Text(
                                    text = "Claim:",
                                    fontWeight = FontWeight.SemiBold,
                                    fontSize = 14.sp,
                                    color = Color(0xFF4A84D4)
                                )
                                Text(
                                    text = data.claim,
                                    fontSize = 15.sp,
                                    modifier = Modifier.padding(bottom = 6.dp)
                                )

                                // Sources
                                if (data.sources.isNotEmpty()) {
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text(
                                        text = "Sources:",
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 14.sp
                                    )
                                    Spacer(modifier = Modifier.height(4.dp))
                                    data.sources.forEach { source ->
                                        Column(modifier = Modifier.padding(bottom = 6.dp)) {
                                            Text(
                                                text = source.title,
                                                fontSize = 14.sp,
                                                fontWeight = FontWeight.Medium
                                            )
                                            Text(
                                                text = source.url,
                                                fontSize = 12.sp,
                                                color = Color(0xFF186BDE),
                                                modifier = Modifier.clickable {
                                                    val intent = Intent(
                                                        Intent.ACTION_VIEW,
                                                        Uri.parse(source.url)
                                                    )
                                                    context.startActivity(intent)
                                                }
                                            )
                                        }
                                    }
                                } else {
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text(
                                        text = "No sources found.",
                                        fontSize = 14.sp,
                                        fontStyle = FontStyle.Italic
                                    )
                                }
                            }
                        }
                    }



                }
            }
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

                    // Search Icon


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
                                if(text.isNotEmpty()){
                                    viewmodel.newPrediction(claim = text)
                                }else{
                                    FancyToast.makeText(
                                        context,
                                        "Please enter something",
                                        FancyToast.LENGTH_LONG,
                                        FancyToast.ERROR,
                                        false
                                    ).show()
                                }
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
                            text = "Upload News",  // Your desired text
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
