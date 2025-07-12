package org.example.project.Presentation.Navigation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import org.example.project.Presentation.Screens.AudioUploadScreen
import org.example.project.Presentation.Screens.HomeScreenUI
import org.example.project.Presentation.Screens.ImageUploadScreen
import org.example.project.Presentation.Screens.VideoUploadScreen
import org.example.project.Presentation.ViewModel.MyViewModel
import org.example.project.data.Repository.RepositoryImpl
import org.example.project.domain.UseCase.UploadAudioToDeepFakeServerUseCase
import org.example.project.domain.UseCase.UploadImageUseCase
import org.example.project.domain.UseCase.UploadVideoToDeepFakeServerUseCase

@Composable
 fun MainApp() {
    val viewModel = remember {
        MyViewModel(uploadVideoToDeepFakeServerUseCase = UploadVideoToDeepFakeServerUseCase(repository = RepositoryImpl()),
            uploadAudioToDeepFakeServerUseCase = UploadAudioToDeepFakeServerUseCase(repository = RepositoryImpl()),
            uploadImageUseCase = UploadImageUseCase(repository = RepositoryImpl())
        )

    }
    val navController= rememberNavController()
    NavHost(navController = navController, startDestination = HOMESCREEN) {
        composable<HOMESCREEN> {
                 //ScreenA(navController = navController)
            HomeScreenUI(navController = navController)

        }
        composable<VIDEOUPLOADSCREEN> {
            VideoUploadScreen(viewModel = viewModel)

        }
        composable<AUDIOUPLOADSCREEN> {
            AudioUploadScreen(viewModel = viewModel)
        }
        composable<IMAGEUPLOADSCREEN> {
            ImageUploadScreen(viewModel = viewModel)

        }
    }

}