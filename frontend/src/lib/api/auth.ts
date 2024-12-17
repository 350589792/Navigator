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
    const params = new URLSearchParams();
    params.append('username', data.email);
    params.append('password', data.password);
    params.append('grant_type', 'password');
    params.append('scope', '');

    const tokenResponse = await api.post<TokenResponse>(
      '/auth/token',
      params,
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      }
    );

    // Set the token for subsequent requests
    const token = tokenResponse.data.access_token;
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;

    // Fetch user data
    const userResponse = await api.get<User>('/users/me');

    return {
      token,
      user: userResponse.data
    };
  },

  register: async (data: RegisterData): Promise<AuthResponse> => {
    // Register the user first
    await api.post<User>('/auth/register', data);
    // Then login to get the token and user data
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
};

export type { AuthResponse };
