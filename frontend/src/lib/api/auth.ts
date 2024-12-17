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
    const formData = new FormData();
    formData.append('username', data.email);
    formData.append('password', data.password);

    const tokenResponse = await api.post<TokenResponse>(
      '/api/v1/auth/login',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    );

    // Set the token for subsequent requests
    const token = tokenResponse.data.access_token;
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;

    // Fetch user data
    const userResponse = await api.get<User>('/api/v1/users/me');

    return {
      token,
      user: userResponse.data
    };
  },

  register: async (data: RegisterData): Promise<AuthResponse> => {
    await api.post<User>('/api/v1/auth/register', data);
    return auth.login({ email: data.email, password: data.password });
  },

  resetPassword: async (email: string): Promise<void> => {
    await api.post('/api/v1/auth/reset-password', { email });
  },

  setNewPassword: async (token: string, password: string): Promise<void> => {
    await api.post('/api/v1/auth/set-new-password', { token, password });
  },

  getCurrentUser: async (): Promise<User> => {
    const response = await api.get<User>('/api/v1/users/me');
    return response.data;
  },
};

export type { AuthResponse };
