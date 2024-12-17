import * as React from 'react';
import { useState } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { useLanguage } from '@/contexts/LanguageContext';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useAuth } from '@/contexts/AuthContext';
import { LanguageSwitcher } from '@/components/LanguageSwitcher';

export default function ResetPassword() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { requestPasswordReset, resetPassword } = useAuth();
  const { t } = useLanguage();
  const [email, setEmail] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  const token = searchParams.get('token');

  const handleRequestReset = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    try {
      await requestPasswordReset(email);
      setSuccess(t('auth.resetPassword.success'));
    } catch (err) {
      setError(t('auth.resetPassword.error'));
    } finally {
      setLoading(false);
    }
  };

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (newPassword !== confirmPassword) {
      setError(t('auth.register.passwordMismatch'));
      setLoading(false);
      return;
    }

    try {
      if (!token) {
        throw new Error('No reset token provided');
      }
      await resetPassword(token, newPassword);
      setSuccess(t('auth.resetPassword.success'));
      setTimeout(() => navigate('/login'), 2000);
    } catch (err) {
      setError(t('auth.resetPassword.error'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <LanguageSwitcher />
      <Card className="w-full max-w-md">
        <CardHeader>
          <h1 className="text-2xl font-bold text-center">{t('auth.resetPassword.title')}</h1>
          <p className="text-center text-gray-500">
            {token ? t('auth.resetPassword.enterNew') : t('auth.resetPassword.enterEmail')}
          </p>
        </CardHeader>
        <CardContent>
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {success && (
            <Alert className="mb-4">
              <AlertDescription>{success}</AlertDescription>
            </Alert>
          )}

          {token ? (
            <form onSubmit={handleResetPassword} className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">{t('auth.resetPassword.newPassword')}</label>
                <Input
                  type="password"
                  placeholder={t('auth.resetPassword.newPasswordPlaceholder')}
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  required
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">{t('auth.resetPassword.confirmPassword')}</label>
                <Input
                  type="password"
                  placeholder={t('auth.resetPassword.confirmPasswordPlaceholder')}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                />
              </div>

              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? t('auth.resetPassword.loading') : t('auth.resetPassword.submit')}
              </Button>
            </form>
          ) : (
            <form onSubmit={handleRequestReset} className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">{t('auth.resetPassword.email')}</label>
                <Input
                  type="email"
                  placeholder={t('auth.resetPassword.emailPlaceholder')}
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>

              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? t('auth.resetPassword.loading') : t('auth.resetPassword.submit')}
              </Button>
            </form>
          )}

          <p className="text-center text-sm mt-4">
            {t('auth.resetPassword.backToLogin')}{' '}
            <Link to="/login" className="text-primary hover:underline">
              {t('auth.login.submit')}
            </Link>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
