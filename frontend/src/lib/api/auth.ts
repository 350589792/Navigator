import { api } from './base';
import type { User, LoginData, RegisterData } from '@/types/api';

interface TokenResponse {
  access_token: string;
  token_type: string;
}

interface AuthResponse {
  token: string;
  user: User;
}

export const auth = {
  login: async (data: LoginData): Promise<AuthResponse> => {
    const formData = new URLSearchParams();
    formData.append('username', data.email);
    formData.append('password', data.password);
    formData.append('grant_type', 'password');

    const tokenResponse = await api.post<TokenResponse>(
      '/auth/login',
      formData.toString(),
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      }
    );

    // Set the token for subsequent requests
    const token = tokenResponse.data.access_token;
    localStorage.setItem('token', token);
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;

    // Fetch user data
    const userResponse = await api.get<User>('/users/me');

    return {
      token,
      user: userResponse.data
    };
  },

  register: async (data: RegisterData): Promise<AuthResponse> => {
    await api.post<User>('/auth/register', data);
    return auth.login({ email: data.email, password: data.password });
  },

  resetPassword: async (email: string): Promise<void> => {
    await api.post('/auth/reset-password', { email });
  },

  setNewPassword: async (token: string, password: string): Promise<void> => {
    await api.post('/auth/set-new-password', { token, password });
  },

  getCurrentUser: async (): Promise<User> => {
    const response = await api.get<User>('/users/me');
    return response.data;
  },

  refreshToken: async (): Promise<TokenResponse | null> => {
    try {
      const response = await api.post<TokenResponse>(
        '/auth/refresh-token',
        {},
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.data && response.data.access_token) {
        localStorage.setItem('token', response.data.access_token);
        api.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`;
        return response.data;
      }
      return null;
    } catch (error) {
      console.error('Error refreshing token:', error);
      localStorage.removeItem('token');
      throw error;
    }
  },
};

export type { AuthResponse, TokenResponse };
