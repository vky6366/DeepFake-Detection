
import axios from 'axios';

const API_BASE_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:5000'
  : `http://${window.location.hostname}:5000`;

const instance = axios.create({
  baseURL: API_BASE_URL,
});

export default instance;