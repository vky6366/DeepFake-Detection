package org.example.project.di

import org.example.project.Presentation.ViewModel.MyViewModel
import org.example.project.data.Repository.RepositoryImpl
import org.example.project.domain.UseCase.UploadVideoToDeepFakeServerUseCase
import org.koin.core.context.startKoin

import org.koin.dsl.module

val module = module {
    factory{ RepositoryImpl() }
    factory { UploadVideoToDeepFakeServerUseCase (repository = get()) }
    factory { MyViewModel(uploadVideoToDeepFakeServerUseCase = get()) }
}

fun main() {
    // ✅ Start Koin
    startKoin {
        modules(module) // 👈 Define this below
    }

}
