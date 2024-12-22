import { useState } from 'react'
import { Scale, Book, FileText } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { ScrollArea } from "@/components/ui/scroll-area"

interface Law {
  law_name: string
  article_number: string
  content: string
}

interface SimilarCase {
  title: string
  summary: string
  similarity_score: number
}

interface AnalysisResult {
  relevant_laws: Law[]
  similar_cases: SimilarCase[]
}

function App() {
  const [caseText, setCaseText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)

  const analyzeCaseText = async () => {
    if (!caseText.trim()) {
      setError('请输入案例内容')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/analyze_case`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ case_text: caseText }),
      })

      if (!response.ok) {
        throw new Error('分析请求失败')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError('分析过程中出现错误，请稍后重试')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-gray-900">交通事故案例分析系统</h1>
          <p className="text-gray-500">输入案例详情，获取相关法条参考和类似案例</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              案例输入
            </CardTitle>
            <CardDescription>请详细描述交通事故案例的具体情况</CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea
              placeholder="请输入案例详情..."
              className="min-h-[200px]"
              value={caseText}
              onChange={(e) => setCaseText(e.target.value)}
            />
            <div className="mt-4 flex justify-end">
              <Button
                onClick={analyzeCaseText}
                disabled={loading}
              >
                {loading ? '分析中...' : '开始分析'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {error && (
          <Alert variant="destructive">
            <AlertTitle>错误</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <Card className="h-full shadow-lg">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-xl">
                  <Scale className="w-6 h-6" />
                  相关法条
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[450px]">
                  <div className="space-y-4">
                    {result.relevant_laws.map((law, index) => (
                      <div key={index} className="p-4 rounded-lg border bg-gray-50">
                        <h3 className="font-semibold text-lg">
                          {law.law_name} {law.article_number}
                        </h3>
                        <p className="mt-2 text-gray-600">{law.content}</p>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>

            <Card className="h-full shadow-lg">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-xl">
                  <Book className="w-6 h-6" />
                  类似案例
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[450px]">
                  <div className="space-y-4">
                    {result.similar_cases.map((case_, index) => (
                      <div key={index} className="p-4 rounded-lg border bg-gray-50">
                        <div className="flex justify-between items-start">
                          <h3 className="font-semibold text-lg">{case_.title}</h3>
                          <span className="text-sm text-gray-500">
                            相似度: {(case_.similarity_score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="mt-2 text-gray-600">{case_.summary}</p>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
