//import React, { Component } from 'react'; // JavaScript로 작성된 React framework를 가져옴
import { 
  TouchableOpacity, 
  StyleSheet, 
  View, Button, 
  Text, 
  SafeAreaView,
  Image, 
  ScrollView, 
  StatusBar } from 'react-native';
import { createStackNavigator, CardStyleInterpolators } from '@react-navigation/stack';


const MainScreen = ({ navigation }) => {
  const button1 = "진단하기"
  const button2 = "진단결과"
  return (
    // SafeAreaView는 ios전용 -> View
    <SafeAreaView style={styles.container}> 
        {/* <Button 
          title={button2}
          color= '#f8f8ff'
          onPress={()=> navigation.navigate('Album')}>
        </Button> */}
        
      <View>
        <TouchableOpacity 
          style={styles.touchableopacity}
          title={button1}
          color= '#f8f8ff'
          // onPress={()=>navigation.push('CameraScreen')}도 가능
          onPress={()=> navigation.navigate('Camera')}>
          <Text style={styles.text}>
            진단하기
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.button_bottom}>
        <Button 
          title={button2}
          color= '#f8f8ff'
          onPress={()=> navigation.navigate('Album')}>
        </Button>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  header:{
    headerTintiColor: '#fff',
    headerStyle: {backgroundColor:'tomato'},
    headerTitleStyle: {
      fontWeight: 'bold',
      fontSize: 20,
    },
  },
  album:{
    headerTintiColor: '#fff',
    headerStyle: {
      backgroundColor:'tomato'
    },
    headerTitleStyle: {
      fontWeight: 'bold',
      fontSize: 20,
    },
    CardStyleInterpolators: CardStyleInterpolators.forRevealFromBottomAndroid
  },
  container: {
    flex: 1,
    alignItems:'center',
    justifyContent:'center',
    //marginHorizontal:16,
    //padding: 20,
    //margin: 10,
    backgroundColor: '#fffacd'
  },
  button_top: {
    flex: 0.07,
    width: 200,
    //height: 10,
    backgroundColor:"#008000",
    borderWidth: 2,
    borderRadius: 20,
    //borderTopLeftRadius: 20,
    //borderTopRightRadius: 20,
    //borderBottomLeftRadius: 20,
    //borderBottomRightRadius: 20,
    padding: 0,
    margin: 50,
  },
  button_bottom: {
    flex: 0.07,
    width: 200,
    //height: 10,
    backgroundColor:"#008000",
    borderWidth: 2,
    borderRadius: 20,
    //borderTopLeftRadius: 20,
    //borderTopRightRadius: 20,
    //borderBottomLeftRadius: 20,
    //borderBottomRightRadius: 20,
    padding: 0,
    margin: 50,
  },
  text: {
    color: '#fff',
    fontWeight: 'bold',
    textAlign: 'center'
  },
  touchableopacity: {
    width: 200,
    borderRadius: 20,
    backgroundColor: "#008000",//'#14274e',
    borderWidth: 2,
    //flexDirection: 'center',
    justifyContent: 'center',
    alignItems: 'center',
    height: 42,
  },
  camera:{
    flex:1,
  },
  cameraButtonContainer:{
    flex:1,
    flexDirection:'row',
    backgroundColor:'transparent',
    margin:64,
  },
  camerascreen_button: {
    width: 130,
    borderRadius: 4,
    backgroundColor: '#14274e',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    height: 40
  },
  camerascreen_text: {
    color: '#fff',
    fontWeight: 'bold',
    textAlign: 'center'
  },
});

export default MainScreen;