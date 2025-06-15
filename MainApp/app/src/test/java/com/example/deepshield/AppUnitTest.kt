package com.example.deepshield

import com.example.deepshield.data.UseCases.GetAllSongUseCase
import com.example.deepshield.data.UseCases.GetGradCamUseCase
import com.example.deepshield.data.UseCases.NewsPredictionUseCase
import com.example.deepshield.data.repoIMPL.TestRepo.FakeRepository
import com.example.deepshield.data.repoIMPL.TestRepo.FakeSongRepository
import com.example.deepshield.domain.StateHandling.ApiResult
import com.example.deepshield.domain.StateHandling.ResultState
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.Assert.*

class AppUnitTest {
    private lateinit var fakeRepository: FakeRepository
    private lateinit var gradCamUseCase: GetGradCamUseCase
    private lateinit var newsPredictionUseCase: NewsPredictionUseCase
    private lateinit var claim: String
    private lateinit var fakeSongRepository: FakeSongRepository
    private lateinit var getAllSongUseCase: GetAllSongUseCase

    @Before
    fun startUp(){
        fakeRepository= FakeRepository()
        gradCamUseCase = GetGradCamUseCase(repository = fakeRepository)
        newsPredictionUseCase = NewsPredictionUseCase(repository = fakeRepository)
        claim = "Donald Trump is president of USA"
        fakeSongRepository = FakeSongRepository()
        getAllSongUseCase = GetAllSongUseCase(repository = fakeSongRepository)

    }

    @After
    fun tearDown(){
        //Auto Garbage collector

    }

    @Test
    fun `get grad cam from server TestCase`() = runBlocking{
        val result =gradCamUseCase.execute().first()
        val report = result is ApiResult.Success
        assertTrue(report)
        val data = (result as ApiResult.Success).data
        val listOfString = listOf<String>("hello1","hello2","hello3")
        assertEquals(listOfString,data.focused_regions)

    }

    @Test
    fun `news Prediction Test Case`() = runBlocking{

        val result = newsPredictionUseCase.invoke(claim =claim ).first()
        val report = result is ApiResult.Success
        assertTrue(report)
        val data = result as ApiResult.Success
        assertEquals(claim,data.data.claim)

    }

    @Test
    fun `get All Songs Test Case`() = runBlocking{
        val result = getAllSongUseCase.invoke().first()
        val report = result is ResultState.Success
        assertTrue(report)
        val data = result as ResultState.Success
        assertEquals("123",data.data[0].id)
        assertEquals("124",data.data[1].id)
        
    }



}