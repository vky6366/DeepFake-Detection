// API wrapper
// only one time the api address need to change no need to change again and again
// axios is used for the newtwork calling
import axios from "axios";

// Create axios instance
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ||"http://localhost:5000",
  withCredentials: false,
});

// Helper for file uploads (multipart/form-data)
export const uploadFile = (path, formData, config = {}) =>
  api.post(path, formData, {
    ...config,
    headers: {
      ...(config.headers || {}),
      "Content-Type": "multipart/form-data",
    },
  });

// Simple GET wrapper
export const get = (path, params) => api.get(path, { params });

export default api;
