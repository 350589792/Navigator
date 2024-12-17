import { Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { LanguageProvider } from './contexts/LanguageContext';
import { Navigation } from './components/Navigation';
import LoginPage from './pages/auth/LoginPage';
import RegisterPage from './pages/auth/RegisterPage';
import ResetPassword from './pages/auth/ResetPassword';
import DomainSelection from './pages/dashboard/DomainSelection';
import EmailPreferences from './pages/dashboard/EmailPreferences';
import ReportViewer from './pages/dashboard/ReportViewer';
import UserManagement from './pages/admin/UserManagement';
import DomainConfig from './pages/admin/DomainConfig';
import LLMConfig from './pages/admin/LLMConfig';
import LogViewer from './pages/admin/LogViewer';
import IntroductionPage from './pages/introduction/IntroductionPage';

function App() {
  return (
    <LanguageProvider>
      <AuthProvider>
        <div className="min-h-screen bg-gray-50">
          <Navigation />
          <div className="py-6">
            <Routes>
              {/* Public Routes */}
              <Route path="/auth/login" element={<LoginPage />} />
              <Route path="/auth/register" element={<RegisterPage />} />
              <Route path="/auth/reset-password" element={<ResetPassword />} />

              {/* Dashboard Routes */}
              <Route path="/dashboard" element={<DomainSelection />} />
              <Route path="/dashboard/email" element={<EmailPreferences />} />
              <Route path="/dashboard/reports" element={<ReportViewer />} />

              {/* Admin Routes */}
              <Route path="/admin/users" element={<UserManagement />} />
              <Route path="/admin/domains" element={<DomainConfig />} />
              <Route path="/admin/llm" element={<LLMConfig />} />
              <Route path="/admin/logs" element={<LogViewer />} />

              {/* Landing Page */}
              <Route path="/" element={<IntroductionPage />} />
            </Routes>
          </div>
        </div>
      </AuthProvider>
    </LanguageProvider>
  );
}

export default App;
