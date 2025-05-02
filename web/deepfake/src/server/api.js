import axios from 'axios';
  
const Api = axios.create({
    baseURL: "http://127.0.0.1:5000",
    // endURL: ENDPOINT
});

export default Api;



