import { useState } from 'react'
import { Button } from "./components/ui/button"
import { Textarea } from "./components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card"
import { ScrollArea } from "./components/ui/scroll-area"
import { useToast } from "./components/ui/use-toast"
import { Toaster } from "./components/ui/toaster"

interface SimilarCase {
  content: string;
  similarity_score: number;
}

interface ApiResponse {
  relevant_laws: string[];
  similar_cases: SimilarCase[];
}

function App() {
  const [caseInput, setCaseInput] = useState('')
  const [laws, setLaws] = useState<string[]>([])
  const [similarCases, setSimilarCases] = useState<SimilarCase[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const { toast } = useToast()

  const handleAnalysis = async () => {
    setIsAnalyzing(true)
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/analyze_case`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: caseInput }),
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data: ApiResponse = await response.json()
      if (!data.relevant_laws || !data.similar_cases) {
        throw new Error('服务器返回数据格式错误')
      }
      
      setLaws(data.relevant_laws)
      setSimilarCases(data.similar_cases)
    } catch (error) {
      console.error('Analysis failed:', error)
      if (error instanceof Error) {
        if (error.message.includes('HTTP error')) {
          toast({
            variant: "destructive",
            title: "错误",
            description: '服务器连接失败，请稍后重试'
          })
        } else if (error.message.includes('数据格式')) {
          toast({
            variant: "destructive",
            title: "错误",
            description: '系统处理出错，请联系管理员'
          })
        } else {
          toast({
            variant: "destructive",
            title: "错误",
            description: '分析案例失败，请稍后重试'
          })
        }
      } else {
        toast({
          variant: "destructive",
          title: "错误",
          description: '网络连接失败，请检查网络后重试'
        })
      }
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <h1 className="text-3xl font-bold text-center text-gray-900">
          交通事故案例分析系统
        </h1>
        
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>案例输入</CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="请输入交通事故案例详情..."
                className="min-h-[200px]"
                value={caseInput}
                onChange={(e) => setCaseInput(e.target.value)}
              />
              <Button 
                className="mt-4 w-full"
                onClick={handleAnalysis}
                disabled={!caseInput || isAnalyzing}
              >
                {isAnalyzing ? '分析中...' : '开始分析'}
              </Button>

            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>相关法条</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[400px]">
                  {laws.length > 0 ? (
                    <div className="space-y-4">
                      {laws.map((law, index) => (
                        <div key={index} className="p-4 bg-white rounded-lg border">
                          {law}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center text-gray-500">
                      暂无相关法条
                    </div>
                  )}
                </ScrollArea>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>类似案例</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[400px]">
                  {similarCases.length > 0 ? (
                    <div className="space-y-4">
                      {similarCases.map((case_, index) => (
                        <div key={index} className="p-4 bg-white rounded-lg border">
                          <div>
                            <div className="mb-2">{case_.content}</div>
                            <div className="text-sm text-gray-500">
                              相似度: {(case_.similarity_score * 100).toFixed(2)}%
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center text-gray-500">
                      暂无类似案例
                    </div>
                  )}
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
      <Toaster />
    </div>
  )
}

export default App
