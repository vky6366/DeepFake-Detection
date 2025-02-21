package com.example.deepshield.presentation.Navigation

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.toRoute
import com.example.deepshield.presentation.Screens.DeepFakeVideoOutput
import com.example.deepshield.presentation.Screens.DeepFakeVideoScreen
import com.example.deepshield.presentation.Screens.HomeScreen
import com.example.deepshield.presentation.Screens.VideoProcessingScreen
import com.example.deepshield.presentation.Screens.VideoScreenSelector
import com.example.deepshield.presentation.Screens.VideoUploadingScreenBin
import com.example.deepshield.presentation.viewModel.MyViewModel

//@Composable
//fun MyApp(viewModel: MyViewModel = hiltViewModel()) {
//    val navController= rememberNavController()
//    NavHost(navController = navController, startDestination =HOMESCREEN ) {
//        composable<HOMESCREEN> {
//            DeepFakeVideoScreen(navController = navController)
//        }
//        composable<DEEPFAKEVIDEOOUTPUTSCREEN> {backstack->
//            val data:DEEPFAKEVIDEOOUTPUTSCREEN = backstack.toRoute()
//            DeepFakeVideoOutput(message = data.message, prediction = data.prediction, score = data.score)
//        }
//        composable<VIDEOUPLOADSCREENROUTE> {backstack->
//            val data:VIDEOUPLOADSCREENROUTE = backstack.toRoute()
//            VideoUploadingScreenBin(videoUri = data.videoUri)
//        }
//
//    }
//}

@Composable
fun MyApp() {
    val navController= rememberNavController()
    NavHost(navController = navController, startDestination =VIDEOSELECTIONSCREEN ) {
        composable<VIDEOSELECTIONSCREEN> {
            VideoScreenSelector(navController = navController)
        }
        composable<VIDEOUPLOADSCREENROUTE> {backstack->
            val data:VIDEOUPLOADSCREENROUTE = backstack.toRoute()
            VideoUploadingScreenBin(videoUri = data.videoUri, imageUri = data.imageUri, navController = navController)
        }
        composable<VIDEOPROCESSINGSCREEN> {backstackdata->
            val data:VIDEOPROCESSINGSCREEN =backstackdata.toRoute()
            VideoProcessingScreen(videoUri = data.videoUri, imageUri = data.imageUri, navController = navController)

        }


    }

}