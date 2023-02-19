import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.IOException

fun main() {
    val client = OkHttpClient()

    val modelFile = File("<path-to-checkpoint-file>.ckpt")
    val modelBytes = modelFile.readBytes()

    val requestUrl = "https://example.com/api/predict"
    val requestBody = "<input-data>".toRequestBody()
    val requestHeaders = mapOf(
        "Content-Type" to "application/json",
        "Authorization" to "Bearer <access-token>"
    )

    val request = Request.Builder()
        .url(requestUrl)
        .post(modelBytes.toRequestBody())
        .headers(requestHeaders.toHeaders())
        .build()

    try {
        val response = client.newCall(request).execute()
        if (response.isSuccessful) {
            val responseBody = response.body?.string()
            println(responseBody)
        } else {
            println("Error: ${response.code} - ${response.message}")
        }
    } catch (e: IOException) {
        println("Error making request: ${e.message}")
    }
}
