package com.example.deepshield.presentation.Navigation

import androidx.compose.runtime.Composable
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.deepshield.presentation.Screens.HomeScreen
import com.example.deepshield.presentation.viewModel.MyViewModel

@Composable
fun MyApp(viewModel: MyViewModel = hiltViewModel()) {
    val navController= rememberNavController()
    NavHost(navController = navController, startDestination =HOMESCREEN ) {
        composable<HOMESCREEN> {
            HomeScreen()
        }

    }
}