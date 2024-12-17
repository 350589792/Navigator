import { Button } from "@/components/ui/button";
import { useLanguage } from "@/contexts/LanguageContext";

export function LanguageSwitcher() {
  const { language, setLanguage } = useLanguage();

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => setLanguage(language === 'zh' ? 'en' : 'zh')}
      className="fixed top-4 right-4 z-50"
      aria-label={language === 'zh' ? 'Switch to English' : '切换到中文'}
    >
      {language === 'zh' ? 'English' : '中文'}
    </Button>
  );
}
