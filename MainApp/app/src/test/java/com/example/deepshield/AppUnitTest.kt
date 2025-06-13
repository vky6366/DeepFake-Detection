package com.example.deepshield

import com.example.deepshield.data.UseCases.GetGradCamUseCase
import com.example.deepshield.data.UseCases.NewsPredictionUseCase
import com.example.deepshield.data.repoIMPL.TestRepo.FakeRepository
import com.example.deepshield.domain.StateHandling.ApiResult
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

    @Before
    fun startUp(){
        fakeRepository= FakeRepository()
        gradCamUseCase = GetGradCamUseCase(repository = fakeRepository)
        newsPredictionUseCase = NewsPredictionUseCase(repository = fakeRepository)
        claim = "Donald Trump is president of USA"

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


}