import axios from "axios";

const BASE_URL = "http://127.0.0.1:5000"; 

const Api = axios.create({
    baseURL: BASE_URL,
    
});

export default Api;